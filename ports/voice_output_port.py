# -*- coding: utf-8 -*-
"""
ports/voice_output_port.py
============================
Contrat pour "dire un texte à voix haute". L'adapter (RVCStreamingVoiceAdapter)
cache tout l'asynchronisme interne (Edge-TTS, décodage MP3, streaming SOLA) --
de l'extérieur, `speak()` est un simple appel bloquant : "dis ce texte, puis
reviens quand c'est fini de jouer".
"""
from abc import ABC, abstractmethod


class VoiceOutputPort(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """Convertit `text` en voix (Miku/BT-7274) et le joue. Bloquant."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Remet à zéro l'état interne du streamer (buffers de contexte SOLA).
        À appeler entre deux réponses indépendantes, pour ne pas mélanger
        le contexte audio de la phrase précédente avec la nouvelle.
        """
        raise NotImplementedError
