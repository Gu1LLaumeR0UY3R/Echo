# -*- coding: utf-8 -*-
"""
gui/visualization_relay.py
=============================
Problème d'ordre de construction : `RVCStreamingVoiceAdapter` a besoin
d'un `AudioVisualizationPort` dès sa création, mais le widget
`WaveformVisualizer` a besoin d'un widget parent Tkinter -- donc de
`MainWindow`, qui elle-même est généralement construite APRÈS le service
vocal dans main.py (EchoService, puis MainWindow(echo_service, ...)).

Solution : ce relay est un `AudioVisualizationPort` valide qu'on peut
créer immédiatement, AVANT tout widget. Tant qu'aucune cible n'est
attachée, il ne fait rien (les blocs sont juste ignorés -- l'audio n'est
jamais impacté). Une fois `MainWindow` construite et son
`WaveformVisualizer` créé, on appelle `relay.attach(waveform)` et le
relay se met à transmettre chaque bloc au widget réel.

USAGE (dans main.py) :
    from gui.visualization_relay import VisualizationRelay

    relay = VisualizationRelay()
    voice = RVCStreamingVoiceAdapter(..., visualizer=relay)
    echo_service = EchoService(voice, ...)
    window = MainWindow(echo_service, speech_adapter, visualizer_relay=relay)
    window.mainloop()
"""
from __future__ import annotations

import numpy as np

from ports.audio_visualization_port import AudioVisualizationPort


class VisualizationRelay(AudioVisualizationPort):
    def __init__(self):
        self._target: "AudioVisualizationPort | None" = None

    def attach(self, target: AudioVisualizationPort) -> None:
        """Appelé une fois le widget WaveformVisualizer construit (dans MainWindow.__init__)."""
        self._target = target

    def push_samples(self, samples: np.ndarray) -> None:
        target = self._target
        if target is not None:
            target.push_samples(samples)
