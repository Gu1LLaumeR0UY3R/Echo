# -*- coding: utf-8 -*-
"""
Adaptateur entre SOLAStreamer et le vrai VoiceEngine (engine.py + pipeline.py
+ hubert_engine.py + rmvpe_engine.py, fichiers fournis le 13/08/2026).

Principe (identique à ce que fait rtrvc.RVC.infer() dans RVC officiel, mais
avec VOS fonctions à vous, pas les leurs) :

  1. On reçoit la fenêtre complète [contexte passé + bloc courant] à 16kHz.
  2. Pitch (RMVPE) et contenu (HuBERT) sont extraits sur TOUTE cette fenêtre
     — c'est nécessaire pour que HuBERT/RMVPE aient assez de contexte pour
     bien fonctionner sur le début du bloc courant. Oui, ça veut dire qu'on
     recalcule le pitch/contenu du contexte à chaque bloc — c'est le coût du
     streaming, exactement comme dans RVC officiel (d'où le besoin de mesurer
     si ça tient dans le budget de `block_ms` sur votre RTX 4070).
  3. On découpe le résultat pour ne garder QUE les frames correspondant au
     nouveau bloc (skip_head → skip_head+return_length), pas le contexte.
  4. Synthèse avec net_g.infer() (le réseau Miku/BT-7274 chargé par
     VoiceEngine.load_voice_model()).

Réutilise directement voice_engine.pipeline._extract_pitch() (déjà testée et
validée dans votre pipeline.py existant) plutôt que de réécrire la logique
pitch en double — moins de risque de divergence/bug.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from hubert_engine import extract_hubert_features


# Même convention que pipeline.py : une frame HuBERT/pitch = 160 échantillons
# à 16kHz = 10ms. C'est aussi l'unité "zc" de stream_config.py côté 16kHz.
_WINDOW_16K = 160


def make_passthrough_infer_fn(cfg):
    """
    Fonction d'inférence factice : renvoie l'audio d'entrée tel quel
    (rééchantillonné de 16kHz vers cfg.sample_rate), sans conversion de
    voix. Sert à tester le pipeline de streaming (SOLA, blocs, callback
    audio) sans dépendre de vos modèles HuBERT/RMVPE.
    """
    import torchaudio.transforms as tat

    upsampler = None

    def infer_fn(input_wav_16k: torch.Tensor, skip_head: int, return_length: int, f0_method: str):
        nonlocal upsampler
        if upsampler is None:
            upsampler = tat.Resample(
                orig_freq=16000, new_freq=cfg.sample_rate, dtype=torch.float32
            ).to(input_wav_16k.device)
        hop_16k = _WINDOW_16K
        tail = input_wav_16k[-(return_length * hop_16k):]
        return upsampler(tail)

    return infer_fn


def make_lite_engine_infer_fn(
    voice_engine,
    cfg,
    pitch_shift: int = 0,
    index_path: str = "",
    index_rate: float = 0.0,
    protect: float = 0.33,
):
    """
    Branche votre VoiceEngine (engine.py) réel sur l'interface attendue par
    SOLAStreamer.

    Paramètres
    ----------
    voice_engine : engine.VoiceEngine
        Instance déjà créée ET avec un modèle déjà chargé
        (voice_engine.load_voice_model("miku_....pth") appelé AVANT).
    cfg : stream_config.StreamConfig
    pitch_shift : demi-tons (comme le paramètre `pitch_shift` de VoiceEngine.convert)
    index_path / index_rate : retrieval optionnel. Laissé à 0.0 par défaut
        pour la latence — le retrieval ajoute un coût de recherche (faiss)
        à chaque bloc. À activer seulement si la qualité sans retrieval
        n'est pas suffisante, et en remesurant la latence après.
    protect : identique au paramètre `protect` de VoiceEngine.convert.
    """
    pipeline = voice_engine.pipeline
    if pipeline is None:
        raise RuntimeError(
            "voice_engine.load_voice_model(...) doit être appelé AVANT "
            "make_lite_engine_infer_fn() (le pipeline n'est créé qu'au "
            "chargement du modèle)."
        )

    hubert_model = voice_engine.hubert_model
    rmvpe_model = voice_engine.rmvpe_model
    net_g = voice_engine.net_g
    version = voice_engine.version
    has_f0 = voice_engine.has_f0
    device = voice_engine.device
    is_half = voice_engine.config.is_half

    index = index_vectors = None
    if index_path and os.path.exists(index_path) and index_rate != 0:
        import faiss

        index = faiss.read_index(index_path)
        index_vectors = index.reconstruct_n(0, index.ntotal)

    def infer_fn(input_wav_16k: torch.Tensor, skip_head: int, return_length: int, f0_method: str):
        audio_16k = input_wav_16k.view(-1)

        # --- 1. Pitch sur toute la fenêtre (contexte + bloc) ---
        # Réutilise TELLE QUELLE la méthode déjà testée de votre pipeline.py.
        # Elle accepte un torch.Tensor directement (rmvpe.infer_from_audio
        # convertit lui-même si besoin), pas besoin de repasser par numpy.
        pitch_coarse_full = pitch_raw_full = None
        if has_f0:
            f0_coarse, f0_raw = pipeline._extract_pitch(rmvpe_model, audio_16k, pitch_shift)
            pitch_coarse_full = torch.tensor(f0_coarse, device=device).unsqueeze(0).long()
            pitch_raw_full = torch.tensor(f0_raw.astype(np.float32), device=device).unsqueeze(0).float()

        # --- 2. HuBERT sur toute la fenêtre ---
        feats = audio_16k.half() if is_half else audio_16k.float()
        feats = feats.view(1, -1).to(device)
        padding_mask = torch.BoolTensor(feats.shape).to(device).fill_(False)
        feats = extract_hubert_features(hubert_model, feats, version, padding_mask=padding_mask)

        # --- 3. Retrieval optionnel (voir docstring : désactivé par défaut) ---
        if index is not None and index_vectors is not None and index_rate != 0:
            npy = feats[0].detach().cpu().numpy()
            if is_half:
                npy = npy.astype("float32")
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(index_vectors[ix] * np.expand_dims(weight, axis=2), axis=1)
            if is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(device) * index_rate
                + (1 - index_rate) * feats
            )

        # --- 4. HuBERT = 1 frame/20ms -> interpolation vers 1 frame/10ms ---
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        # --- 5. Alignement des longueurs (HuBERT/pitch peuvent différer de
        #        quelques frames selon les arrondis des convolutions) ---
        p_len_full = min(
            feats.shape[1],
            audio_16k.shape[0] // _WINDOW_16K,
            pitch_coarse_full.shape[1] if has_f0 else feats.shape[1],
        )
        feats = feats[:, :p_len_full, :]
        if has_f0:
            pitch_coarse_full = pitch_coarse_full[:, :p_len_full]
            pitch_raw_full = pitch_raw_full[:, :p_len_full]

        # --- 6. On ne garde QUE la portion "nouveau bloc" ---
        end = min(skip_head + return_length, p_len_full)
        start = min(skip_head, end)
        feats = feats[:, start:end, :]
        pitch_coarse = pitch_coarse_full[:, start:end] if has_f0 else None
        pitch_raw = pitch_raw_full[:, start:end] if has_f0 else None

        p_len = feats.shape[1]
        if p_len == 0:
            # Fenêtre encore en warm-up (pas assez de contexte accumulé) :
            # on renvoie du silence de la bonne longueur pour ne pas casser
            # le SOLAStreamer.
            return torch.zeros(
                return_length * cfg.zc, device=device, dtype=torch.float32
            )

        p_len_tensor = torch.tensor([p_len], device=device).long()
        sid_tensor = torch.tensor(0, device=device).unsqueeze(0).long()

        # --- 7. Synthèse ---
        with torch.no_grad():
            if has_f0:
                out = net_g.infer(feats, p_len_tensor, pitch_coarse, pitch_raw, sid_tensor)[0]
            else:
                out = net_g.infer(feats, p_len_tensor, sid_tensor)[0]
            audio_out = out[0, 0].data.float()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_out

    return infer_fn
