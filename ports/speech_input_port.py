# -*- coding: utf-8 -*-
"""
ports/speech_input_port.py
============================
Contrat pour "transcrire ce que dit l'utilisateur au micro", avec
détection automatique de fin de parole (silence) et aperçu en direct de
la transcription pendant l'enregistrement.
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional

# Appelé en continu pendant l'enregistrement avec le texte provisoire
# reconnu jusqu'ici (peut changer/se corriger d'un appel à l'autre, c'est
# normal -- c'est un aperçu, pas le résultat final).
OnPartialCallback = Callable[[str], None]

# Appelé une seule fois, automatiquement, quand un silence a été détecté
# après de la parole -- signale que l'enregistrement s'est arrêté tout
# seul (l'appelant doit alors récupérer le texte final, voir
# stop_recording_and_transcribe).
OnSilenceStopCallback = Callable[[], None]


class SpeechInputPort(ABC):
    @abstractmethod
    def start_recording(
        self,
        on_partial: Optional[OnPartialCallback] = None,
        on_silence_stop: Optional[OnSilenceStopCallback] = None,
    ) -> None:
        """
        Démarre l'enregistrement (non bloquant). S'arrête automatiquement
        tout seul après un silence détecté suivant de la parole (appelle
        alors on_silence_stop), mais reste arrêtable manuellement via
        stop_recording_and_transcribe() à tout moment.
        """
        raise NotImplementedError

    @abstractmethod
    def stop_recording_and_transcribe(self) -> str:
        """
        Arrête l'enregistrement (si pas déjà stoppé automatiquement) et
        retourne le texte final transcrit (chaîne vide si rien compris).
        """
        raise NotImplementedError