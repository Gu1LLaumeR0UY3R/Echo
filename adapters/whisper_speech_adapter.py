# -*- coding: utf-8 -*-
"""
adapters/whisper_speech_adapter.py
=====================================
Implémentation concrète de SpeechInputPort avec faster-whisper (Whisper
optimisé, tourne bien même sur CPU). Choisi plutôt qu'une API cloud pour
rester cohérent avec votre philosophie "aucune dépendance externe" et
éviter d'envoyer votre voix à un tiers.

Par défaut sur CPU (`device="cpu"`) plutôt que GPU : votre RTX 4070 est
déjà partagée entre RVC (HuBERT/RMVPE/synthèse) et potentiellement Qwen --
on a déjà vu que la VRAM est le facteur limitant (cf. l'épisode
qwen2.5:14b qui débordait en CPU/GPU split). Un modèle Whisper "base" en
int8 sur CPU reste largement assez rapide pour quelques secondes de
parole (généralement < 1s de transcription).
"""
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from ports.speech_input_port import SpeechInputPort


class WhisperSpeechAdapter(SpeechInputPort):
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = 16000,
        language: str = "en",
    ):
        print(f"[whisper] Chargement du modèle '{model_size}' ({device}, {compute_type})...")
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type,
            cpu_threads=2,  # limite le nb de threads CPU utilisés par Whisper,
                            # pour ne pas cannibaliser le CPU pendant que RVC tourne
        )
        self.sample_rate = sample_rate
        self.language = language
        self._frames = []
        self._stream = None

    def start_recording(self) -> None:
        self._frames = []

        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            self._frames.append(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32", callback=callback
        )
        self._stream.start()

    def stop_recording_and_transcribe(self) -> str:
        if self._stream is None:
            return ""
        self._stream.stop()
        self._stream.close()
        self._stream = None

        if not self._frames:
            return ""
        audio = np.concatenate(self._frames)
        self._frames = []

        if audio.shape[0] < self.sample_rate * 0.3:
            # Moins de 300ms enregistrées : trop court pour être fiable,
            # évite de renvoyer une transcription farfelue sur du silence.
            return ""

        segments, _ = self.model.transcribe(audio, language=self.language)
        return " ".join(seg.text.strip() for seg in segments).strip()
