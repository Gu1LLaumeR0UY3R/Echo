# -*- coding: utf-8 -*-
"""
pipeline.py
===========
Le "chef d'orchestre" qui enchaîne les 3 briques pour transformer un audio
neutre en audio avec la voix de Miku ou BT-7274 :

  1. (si le modèle le nécessite) extraction du pitch avec RMVPE
  2. extraction du contenu vocal avec HuBERT
  3. synthèse finale avec le réseau de neurones du modèle .pth choisi

Version simplifiée par rapport au RVC officiel : on a retiré la gestion multi-
speaker complexe, le CUDA Graph (optimisation avancée), le support d'autres
méthodes de pitch (pm/fcpe) pour ne garder que RMVPE (la plus fiable dans les
tests qu'on a déjà faits). Le découpage en tronçons pour les longs audios est
conservé car nécessaire pour ne pas saturer la VRAM sur de longues phrases.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from scipy import signal

from hubert_engine import extract_hubert_features

# Filtre passe-haut, retire les fréquences très basses (souffle, bruit de
# fond) avant traitement. Reprend le même filtre que RVC original.
_BUTTER_B, _BUTTER_A = signal.butter(N=5, Wn=48, btype="high", fs=16000)


class Pipeline:
    def __init__(self, target_sample_rate: int, config):
        self.device = config.device
        self.is_half = config.is_half

        # Paramètres de découpage (en secondes, convertis en échantillons).
        self.sr = 16000  # taux d'entrée attendu par HuBERT
        self.window = 160  # taille de fenêtre en échantillons
        self.t_pad = self.sr * config.x_pad
        self.t_pad_tgt = target_sample_rate * config.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * config.x_query
        self.t_center = self.sr * config.x_center
        self.t_max = self.sr * config.x_max

    def _extract_pitch(self, rmvpe_model, audio_padded, pitch_shift_semitones):
        """
        Utilise RMVPE pour détecter la hauteur de voix (f0) à chaque instant,
        puis la convertit dans le format attendu par le réseau de synthèse
        (un indice "coarse" 1-255, plus la valeur f0 brute).
        """
        f0 = rmvpe_model.infer_from_audio(audio_padded, thred=0.03)

        # Comble les silences (f0=0) par interpolation, pour éviter des
        # discontinuités brutales dans la voix générée.
        voiced = f0 != 0
        if voiced.any() and (~voiced).any():
            f0[~voiced] = np.interp(
                np.where(~voiced)[0], np.where(voiced)[0], f0[voiced]
            )

        # Transposition (monter/descendre la voix de X demi-tons).
        f0 *= pow(2, pitch_shift_semitones / 12)
        f0_raw = f0.copy()

        # Conversion en échelle "mel" puis en indice discret 1-255, c'est le
        # format que le réseau de neurones attend en entrée pour le pitch.
        f0_min, f0_max = 50, 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0_mel = 1127 * np.log(1 + f0_raw / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel = np.clip(f0_mel, 1, 255)
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0_raw

    def _synthesize_chunk(
        self, hubert_model, net_g, speaker_id, audio_chunk,
        pitch_coarse, pitch_raw, index, index_vectors, index_rate, version, protect,
    ):
        """
        Traite UN tronçon d'audio (voir pourquoi le découpage existe dans
        `run` ci-dessous) : extrait le contenu vocal, applique éventuellement
        l'index de similarité (retrieval), et fait tourner le réseau de
        synthèse pour produire l'audio final de ce tronçon.
        """
        feats = torch.from_numpy(audio_chunk)
        feats = feats.half() if self.is_half else feats.float()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        feats = extract_hubert_features(
            hubert_model, feats.to(self.device), version, padding_mask=padding_mask
        )

        # "protect" : évite que l'étape de retrieval (index) n'écrase trop le
        # contenu vocal sur les zones non-voisées (consonnes, silences), ce
        # qui produirait des artefacts. On garde une copie du signal "pur".
        has_pitch = pitch_coarse is not None and pitch_raw is not None
        if protect < 0.5 and has_pitch:
            feats_protected = feats.clone()

        # Retrieval : mélange les caractéristiques extraites avec les
        # caractéristiques les plus proches trouvées dans le fichier .index
        # (qui contient des exemples de la vraie voix cible). Ça améliore la
        # fidélité au timbre d'origine.
        if index is not None and index_vectors is not None and index_rate != 0:
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(index_vectors[ix] * np.expand_dims(weight, axis=2), axis=1)
            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        # HuBERT produit une frame toutes les 20ms, le réseau de synthèse en
        # attend une toutes les 10ms : on double la résolution temporelle.
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and has_pitch:
            feats_protected = F.interpolate(
                feats_protected.permute(0, 2, 1), scale_factor=2
            ).permute(0, 2, 1)

        p_len = audio_chunk.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if has_pitch:
                pitch_coarse = pitch_coarse[:, :p_len]
                pitch_raw = pitch_raw[:, :p_len]

        if protect < 0.5 and has_pitch:
            protect_mask = pitch_raw.clone()
            protect_mask[pitch_raw > 0] = 1
            protect_mask[pitch_raw < 1] = protect
            protect_mask = protect_mask.unsqueeze(-1)
            feats = feats * protect_mask + feats_protected * (1 - protect_mask)
            feats = feats.to(feats_protected.dtype)

        p_len_tensor = torch.tensor([p_len], device=self.device).long()
        sid_tensor = torch.tensor(speaker_id, device=self.device).unsqueeze(0).long()

        with torch.no_grad():
            if has_pitch:
                out = net_g.infer(feats, p_len_tensor, pitch_coarse, pitch_raw, sid_tensor)[0]
            else:
                out = net_g.infer(feats, p_len_tensor, sid_tensor)[0]
            audio_out = out[0, 0].data.cpu().float().numpy()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_out

    def run(
        self, hubert_model, rmvpe_model, net_g, audio, pitch_shift_semitones,
        file_index_path, index_rate, has_f0, target_sample_rate, protect, version,
    ):
        """
        Point d'entrée principal : prend l'audio complet (potentiellement
        long) et le traite. Les longs audios sont découpés en tronçons aux
        endroits les plus "silencieux" possible, pour éviter de couper au
        milieu d'un mot et pour ne pas saturer la mémoire GPU.
        """
        # Chargement de l'index de similarité (fichier .index), s'il existe.
        index = index_vectors = None
        if file_index_path and os.path.exists(file_index_path) and index_rate != 0:
            index = faiss.read_index(file_index_path)
            index_vectors = index.reconstruct_n(0, index.ntotal)

        audio = signal.filtfilt(_BUTTER_B, _BUTTER_A, audio)

        # Recherche des points de coupe optimaux (zones les plus silencieuses)
        # pour les audios plus longs que ~1 minute (t_max).
        audio_padded_for_search = np.pad(
            audio, (self.window // 2, self.window // 2), mode="reflect"
        )
        cut_points = []
        if audio_padded_for_search.shape[0] > self.t_max:
            energy = np.zeros_like(audio)
            for i in range(self.window):
                energy += np.abs(audio_padded_for_search[i: i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                window_slice = energy[t - self.t_query: t + self.t_query]
                cut_points.append(
                    t - self.t_query + int(np.argmin(window_slice))
                )

        audio_padded = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_padded.shape[0] // self.window

        pitch_coarse = pitch_raw = None
        if has_f0:
            pitch_coarse, pitch_raw = self._extract_pitch(
                rmvpe_model, audio_padded, pitch_shift_semitones
            )
            pitch_coarse = pitch_coarse[:p_len]
            pitch_raw = pitch_raw[:p_len].astype(np.float32)
            pitch_coarse = torch.tensor(pitch_coarse, device=self.device).unsqueeze(0).long()
            pitch_raw = torch.tensor(pitch_raw, device=self.device).unsqueeze(0).float()

        speaker_id = 0  # un seul locuteur par modèle dans nos fichiers
        audio_chunks = []
        start = 0
        cut = None
        for cut in cut_points:
            cut = cut // self.window * self.window
            chunk = audio_padded[start: cut + self.t_pad2 + self.window]
            chunk_pitch_c = (
                pitch_coarse[:, start // self.window: (cut + self.t_pad2) // self.window]
                if has_f0 else None
            )
            chunk_pitch_r = (
                pitch_raw[:, start // self.window: (cut + self.t_pad2) // self.window]
                if has_f0 else None
            )
            result = self._synthesize_chunk(
                hubert_model, net_g, speaker_id, chunk,
                chunk_pitch_c, chunk_pitch_r, index, index_vectors,
                index_rate, version, protect,
            )
            audio_chunks.append(result[self.t_pad_tgt: -self.t_pad_tgt])
            start = cut

        # Dernier tronçon (ou audio entier si pas de découpe nécessaire).
        chunk = audio_padded[start:]
        chunk_pitch_c = pitch_coarse[:, start // self.window:] if has_f0 and cut is not None else pitch_coarse
        chunk_pitch_r = pitch_raw[:, start // self.window:] if has_f0 and cut is not None else pitch_raw
        result = self._synthesize_chunk(
            hubert_model, net_g, speaker_id, chunk,
            chunk_pitch_c, chunk_pitch_r, index, index_vectors,
            index_rate, version, protect,
        )
        audio_chunks.append(result[self.t_pad_tgt: -self.t_pad_tgt])

        audio_out = np.concatenate(audio_chunks)

        # Normalisation finale pour éviter la saturation (clipping).
        peak = np.abs(audio_out).max() / 0.99
        max_int16 = 32768 / peak if peak > 1 else 32768
        audio_out = (audio_out * max_int16).astype(np.int16)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_out
