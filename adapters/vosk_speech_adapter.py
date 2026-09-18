# -*- coding: utf-8 -*-
"""
adapters/vosk_speech_adapter.py
==================================
Implémentation de SpeechInputPort avec Vosk, streaming : le texte est
transcrit progressivement PENDANT que vous parlez (pas juste à la fin),
et l'enregistrement s'arrête tout seul après un silence -- plus besoin de
recliquer pour arrêter.

Comment la détection de silence fonctionne : à chaque petit paquet audio
reçu du micro (~toutes les 30-50ms), on mesure son amplitude moyenne. Si
elle dépasse `silence_threshold`, on considère que vous parlez. Si vous
avez parlé au moins `min_speech_duration_s`, PUIS que l'amplitude repasse
sous le seuil pendant `silence_duration_s` d'affilée, on considère que
vous avez fini de parler et on déclenche l'arrêt automatique.

Point technique important : `on_silence_stop` est appelé DEPUIS le thread
audio (callback PortAudio) -- on n'y arrête PAS le flux directement (ce
n'est pas sûr de le faire depuis son propre callback), on se contente de
prévenir l'appelant, qui doit ensuite appeler
`stop_recording_and_transcribe()` depuis un autre thread (c'est ce que
fait gui/main_window.py).
"""
import json
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from ports.speech_input_port import SpeechInputPort, OnPartialCallback, OnSilenceStopCallback


class VoskSpeechAdapter(SpeechInputPort):
    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        silence_threshold: float = 0.015,
        silence_duration_s: float = 1.2,
        min_speech_duration_s: float = 0.4,
    ):
        """
        silence_threshold    : amplitude moyenne en dessous de laquelle
                                c'est considéré comme du silence (0.0-1.0,
                                audio en float32). Si l'arrêt automatique
                                se déclenche trop tôt (coupe vos mots),
                                baissez cette valeur. S'il ne se déclenche
                                jamais (bruit de fond trop présent),
                                montez-la.
        silence_duration_s   : durée de silence continu requise après
                                de la parole avant l'arrêt automatique.
        min_speech_duration_s: durée minimale de parole avant qu'un
                                silence puisse déclencher l'arrêt -- évite
                                un arrêt immédiat sur un bref bruit isolé.
        """
        print(f"[vosk] Chargement du modèle depuis '{model_path}'...")
        self.model = Model(model_path=model_path)
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration_s = silence_duration_s
        self.min_speech_duration_s = min_speech_duration_s

        self._stream: Optional[sd.InputStream] = None
        self._recognizer: Optional[KaldiRecognizer] = None
        self._stopped = threading.Event()
        self._speech_started_at: Optional[float] = None
        self._silence_started_at: Optional[float] = None

    def start_recording(
        self,
        on_partial: Optional[OnPartialCallback] = None,
        on_silence_stop: Optional[OnSilenceStopCallback] = None,
    ) -> None:
        self._recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self._stopped.clear()
        self._speech_started_at = None
        self._silence_started_at = None
        # Vosk a SON PROPRE détecteur de silence interne : quand
        # AcceptWaveform() renvoie True, il "vide" le texte reconnu dans
        # Result() et repart de zéro en interne. Si on ne le récupère pas
        # à ce moment-là, il est perdu -- il faut l'accumuler ici, pas
        # attendre notre propre détection de silence pour tout récupérer
        # d'un coup via FinalResult() (qui serait alors vide).
        self._final_segments: list[str] = []

        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            if self._stopped.is_set():
                return

            chunk = indata[:, 0]
            level = float(np.abs(chunk).mean())
            now = time.monotonic()

            if level > self.silence_threshold:
                if self._speech_started_at is None:
                    self._speech_started_at = now
                self._silence_started_at = None
            elif self._speech_started_at is not None and self._silence_started_at is None:
                self._silence_started_at = now

            audio_int16 = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16)
            if self._recognizer.AcceptWaveform(audio_int16.tobytes()):
                result = json.loads(self._recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    self._final_segments.append(text)
                    if on_partial:
                        on_partial(" ".join(self._final_segments))
            elif on_partial:
                partial = json.loads(self._recognizer.PartialResult())
                text = partial.get("partial", "").strip()
                preview = " ".join(self._final_segments + ([text] if text else []))
                if preview:
                    on_partial(preview)

            speech_long_enough = (
                self._speech_started_at is not None
                and (now - self._speech_started_at) >= self.min_speech_duration_s
            )
            silence_long_enough = (
                self._silence_started_at is not None
                and (now - self._silence_started_at) >= self.silence_duration_s
            )
            if speech_long_enough and silence_long_enough:
                self._stopped.set()  # ignore les paquets audio suivants
                if on_silence_stop:
                    on_silence_stop()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            blocksize=int(self.sample_rate * 0.03),  # ~30ms par paquet, réactif sans surcharger
            callback=callback,
        )
        self._stream.start()

    def stop_recording_and_transcribe(self) -> str:
        if self._stream is None:
            return ""
        self._stopped.set()
        self._stream.stop()
        self._stream.close()
        self._stream = None

        if self._recognizer is None:
            return ""
        # Récupère ce qui restait "en cours" (pas encore finalisé par Vosk
        # au moment où on a arrêté), en plus des segments déjà accumulés
        # pendant l'enregistrement.
        final = json.loads(self._recognizer.FinalResult())
        leftover = final.get("text", "").strip()
        if leftover:
            self._final_segments.append(leftover)
        self._recognizer = None
        return " ".join(self._final_segments).strip()