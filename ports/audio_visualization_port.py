# -*- coding: utf-8 -*-
"""
ports/audio_visualization_port.py
====================================
Contrat pour "recevoir des échantillons audio bruts en temps réel, pour
affichage". Volontairement séparé de VoiceOutputPort : la sortie voix
fonctionne déjà et est validée (TEST 1 à 6), on ne touche pas à son
contrat existant. La visualisation est un port additionnel, optionnel,
injecté à côté.

C'est un pattern "observer" : l'adapter audio (RealtimePlayer) POUSSE les
blocs vers ce port au fur et à mesure qu'ils sont joués. Le port ne
retourne rien et ne doit JAMAIS bloquer ni lever d'exception qui remonterait
jusqu'au pipeline audio -- une erreur d'affichage ne doit jamais couper
le son.
"""
from abc import ABC, abstractmethod
import numpy as np


class AudioVisualizationPort(ABC):
    @abstractmethod
    def push_samples(self, samples: np.ndarray) -> None:
        """
        Reçoit un bloc d'échantillons audio float32 mono (typiquement un
        bloc de ~200ms, à la fréquence de streamer/cfg.sample_rate).
        DOIT être non-bloquant (juste empiler dans une queue interne) --
        le rendu visuel se fait ailleurs, sur un thread/timer séparé.
        """
        raise NotImplementedError
