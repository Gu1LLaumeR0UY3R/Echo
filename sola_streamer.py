# -*- coding: utf-8 -*-
"""
SOLAStreamer : le coeur du streaming temps réel pour RVC.

Adapté de RVCStreamEngine.process() dans le repo officiel RVC
(RVCRealtimeVST/worker/rvc_worker.py, récupéré le 12/08/2026), mais
totalement découplé du reste de RVC : on ne dépend PAS de infer.rtrvc.RVC,
ni de fairseq, ni de l'archi VST/mmap Windows. On branche notre propre
fonction d'inférence (infer_fn) qui vient de votre pipeline maison
(hubert_engine.py + rmvpe_engine.py + engine.py).

Principe :
1. On garde un buffer glissant ("input_wav") = [contexte passé] + [bloc courant]
2. On envoie ce buffer complet à l'inférence (infer_fn) à CHAQUE bloc — le
   contexte passé sert à ce que HuBERT/RMVPE aient assez d'historique.
3. On ne garde/joue que la fin du résultat (le "nouveau" morceau), mais on
   applique un recouvrement SOLA (Synchronized OverLap-Add) : on cherche
   le meilleur point d'alignement entre la fin du bloc précédent et le
   début du bloc courant pour fondre les deux sans "clic" ni discontinuité
   de pitch.

C'est ça qui remplace le "chunked" actuel (découper une phrase entière déjà
générée) par du vrai streaming (jouer pendant que ça continue à être généré).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat

from stream_config import StreamConfig


class SOLAStreamer:
    def __init__(self, cfg: StreamConfig, infer_fn, device: str = "cuda"):
        """
        Paramètres
        ----------
        cfg : StreamConfig
            Tailles de fenêtres (voir stream_config.py).
        infer_fn : Callable[[torch.Tensor, int, int, str], torch.Tensor]
            Fonction d'inférence à brancher. Signature IDENTIQUE à
            rtrvc.RVC.infer() dans RVC officiel :
                infer_fn(input_wav_16k, skip_head, return_length, f0_method) -> torch.Tensor
            - input_wav_16k : tenseur 1D, la fenêtre [contexte + bloc] rééchantillonnée à 16kHz
            - skip_head     : nb de frames (à 16kHz/zc) à ignorer en tête du résultat
                              (= la portion "contexte" qu'on ne veut pas rejouer)
            - return_length : nb de frames à retourner en sortie
            - f0_method     : "rmvpe" / "fcpe" / "pm"
            Retourne un torch.Tensor 1D à cfg.sample_rate (PAS 16kHz), de
            longueur >= cfg.block_frame + cfg.sola_buffer_frame + cfg.sola_search_frame.
            => C'est ICI qu'on branche votre pipeline.py (hubert_engine + rmvpe_engine).
              Voir infer_adapter.py pour le squelette à compléter.
        device : "cuda" ou "cpu"
        """
        self.cfg = cfg
        self.infer_fn = infer_fn
        self.device = device

        total_frames = cfg.extra_frame + cfg.crossfade_frame + cfg.sola_search_frame + cfg.block_frame
        self.input_wav = torch.zeros(total_frames, device=device, dtype=torch.float32)
        self.input_wav_res = torch.zeros(160 * total_frames // cfg.zc, device=device, dtype=torch.float32)

        self.sola_buffer = torch.zeros(cfg.sola_buffer_frame, device=device, dtype=torch.float32)
        self.sola_den_kernel = torch.ones(1, 1, cfg.sola_buffer_frame, device=device, dtype=torch.float32)
        self.fade_in_window = torch.sin(
            0.5 * np.pi * torch.linspace(0.0, 1.0, steps=cfg.sola_buffer_frame, device=device)
        ) ** 2
        self.fade_out_window = 1 - self.fade_in_window

        self.resampler = tat.Resample(orig_freq=cfg.sample_rate, new_freq=16000, dtype=torch.float32).to(device)

        self._warm = False

    def reset(self):
        """Remet tous les buffers à zéro (à faire entre deux phrases/utterances)."""
        self.input_wav.zero_()
        self.input_wav_res.zero_()
        self.sola_buffer.zero_()

    def push_block(self, audio_block: np.ndarray) -> np.ndarray:
        """
        Envoie un bloc de cfg.block_frame échantillons (float32, à cfg.sample_rate)
        et retourne le bloc converti correspondant (même longueur en échantillons),
        prêt à être envoyé à la sortie audio.
        """
        cfg = self.cfg
        if audio_block.shape[0] != cfg.block_frame:
            raise ValueError(
                f"Bloc attendu de {cfg.block_frame} échantillons, reçu {audio_block.shape[0]}. "
                f"Utilisez AudioBlockizer pour découper vos flux en blocs de la bonne taille."
            )

        # 1) Fenêtre glissante : on décale le buffer et on insère le nouveau bloc à la fin
        self.input_wav[:-cfg.block_frame] = self.input_wav[cfg.block_frame:].clone()
        self.input_wav[-cfg.block_frame:] = torch.from_numpy(audio_block).to(self.device)

        self.input_wav_res[:-cfg.block_frame_16k] = self.input_wav_res[cfg.block_frame_16k:].clone()
        resample_input = self.input_wav[-cfg.block_frame - 2 * cfg.zc:]
        resampled = self.resampler(resample_input)[160:]
        self.input_wav_res[-cfg.block_frame_16k:] = resampled[-cfg.block_frame_16k:]

        # 2) Inférence sur la fenêtre [contexte + bloc courant] à 16kHz
        infer_wav = self.infer_fn(self.input_wav_res, cfg.skip_head, cfg.return_length, cfg.f0_method)
        if not torch.is_tensor(infer_wav):
            infer_wav = torch.as_tensor(infer_wav, device=self.device, dtype=torch.float32)

        # Filet de sécurité : HuBERT/RMVPE peuvent parfois produire quelques
        # frames de moins que prévu (effet de bord des convolutions, plus
        # visible sur certaines phrases/longueurs de texte que d'autres).
        # On complète par du silence plutôt que de planter.
        expected_len = cfg.block_frame + cfg.sola_buffer_frame + cfg.sola_search_frame
        if infer_wav.shape[-1] < expected_len:
            pad = expected_len - infer_wav.shape[-1]
            infer_wav = F.pad(infer_wav, (0, pad))
        elif infer_wav.shape[-1] > expected_len:
            infer_wav = infer_wav[:expected_len]

        # 3) SOLA : on cherche le décalage qui maximise la corrélation normalisée
        #    entre le buffer de fin du bloc précédent (sola_buffer) et le début
        #    du résultat courant. Ça évite les clics dus à un mauvais alignement
        #    de phase entre deux blocs générés indépendamment.
        conv_input = infer_wav[None, None, : cfg.sola_buffer_frame + cfg.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input ** 2, self.sola_den_kernel) + 1e-8)
        sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]))

        infer_wav = infer_wav[sola_offset:]
        infer_wav[: cfg.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: cfg.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[cfg.block_frame: cfg.block_frame + cfg.sola_buffer_frame]

        return infer_wav[: cfg.block_frame].detach().float().cpu().numpy()


class AudioBlockizer:
    """
    Petit utilitaire pour découper un flux audio arbitraire (ex: sortie
    d'Edge-TTS qui arrive par paquets de taille variable) en blocs de taille
    fixe (cfg.block_frame), requis par SOLAStreamer.push_block().

    Usage :
        blk = AudioBlockizer(cfg.block_frame)
        for chunk in flux_audio_variable:      # ex: chunks Edge-TTS décodés en PCM
            for block in blk.feed(chunk):
                converted = streamer.push_block(block)
                jouer(converted)
        for block in blk.flush():              # dernier bloc partiel, complété de silence
            converted = streamer.push_block(block)
            jouer(converted)
    """

    def __init__(self, block_frame: int):
        self.block_frame = block_frame
        self._carry = np.zeros(0, dtype=np.float32)

    def feed(self, samples: np.ndarray):
        self._carry = np.concatenate([self._carry, samples.astype(np.float32)])
        blocks = []
        while len(self._carry) >= self.block_frame:
            blocks.append(self._carry[: self.block_frame])
            self._carry = self._carry[self.block_frame:]
        return blocks

    def flush(self):
        if len(self._carry) == 0:
            return []
        pad = np.zeros(self.block_frame - len(self._carry), dtype=np.float32)
        last = np.concatenate([self._carry, pad])
        self._carry = np.zeros(0, dtype=np.float32)
        return [last]
