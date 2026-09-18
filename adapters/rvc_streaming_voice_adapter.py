# -*- coding: utf-8 -*-
"""
adapters/rvc_streaming_voice_adapter.py
=========================================
Implémentation concrète de VoiceOutputPort. C'est le pont entre le
port abstrait et TOUT ce qu'on a déjà construit et validé (TEST 1 à 6) :
engine.py (VoiceEngine), sola_streamer.py (SOLAStreamer), infer_adapter.py
(make_lite_engine_infer_fn) et tts_streaming_bridge.py (speak_streaming).
Rien de nouveau techniquement ici -- juste emballé derrière le port pour
que le reste de l'app (app/, gui/) n'ait jamais à connaître ces détails.

`speak()` est volontairement une méthode SYNCHRONE (pas de async/await
côté appelant) : elle gère elle-même l'exécution de la coroutine interne
via asyncio.run(). Ça évite à l'interface graphique (Tkinter, qui n'est
pas async-native) d'avoir à gérer une boucle asyncio -- elle appelle juste
`voice.speak(texte)` depuis un thread d'arrière-plan, comme n'importe quel
appel bloquant classique.

-----------------------------------------------------------------------
MODIF VISUALISATION (ajout) :
Nouveau paramètre optionnel `visualizer: AudioVisualizationPort | None`.
S'il est fourni, il est branché sur le RealtimePlayer interne : chaque
bloc audio joué est aussi poussé vers `visualizer.push_samples()`.
Aucun changement de comportement si `visualizer` n'est pas fourni.
-----------------------------------------------------------------------
"""
import asyncio
from typing import Optional

from engine import VoiceEngine
from streaming_config_rvc import StreamingRVCConfig
from stream_config import StreamConfig
from sola_streamer import SOLAStreamer
from infer_adapter import make_lite_engine_infer_fn
from tts_streaming_bridge import speak_streaming, RealtimePlayer
from ports.voice_output_port import VoiceOutputPort
from ports.audio_visualization_port import AudioVisualizationPort


class RVCStreamingVoiceAdapter(VoiceOutputPort):
    def __init__(
        self,
        hubert_pt_path: str,
        rmvpe_pt_path: str,
        voice_model_path: str,
        pitch_shift: int = 0,
        edge_voice: str = "en-US-AriaNeural",
        block_ms: float = 200.0,
        crossfade_ms: float = 40.0,
        extra_ms: float = 500.0,
        visualizer: Optional[AudioVisualizationPort] = None,
    ):
        self.config = StreamingRVCConfig(is_half=True)
        self.voice_engine = VoiceEngine(self.config, hubert_pt_path, rmvpe_pt_path)
        self.voice_engine.load_voice_model(voice_model_path)

        self.cfg = StreamConfig(
            sample_rate=self.voice_engine.target_sr,
            block_ms=block_ms,
            crossfade_ms=crossfade_ms,
            extra_ms=extra_ms,
        )
        infer_fn = make_lite_engine_infer_fn(
            self.voice_engine, self.cfg, pitch_shift=pitch_shift, index_path="", index_rate=0.0
        )
        self.streamer = SOLAStreamer(self.cfg, infer_fn, device=self.config.device)

        self.visualizer = visualizer
        on_block = visualizer.push_samples if visualizer is not None else None
        self.player = RealtimePlayer(self.cfg.sample_rate, self.cfg.block_frame, on_block=on_block)
        self.edge_voice = edge_voice

    def speak(self, text: str) -> None:
        asyncio.run(
            speak_streaming(text, self.streamer, self.cfg, voice=self.edge_voice, player=self.player)
        )
        self.player.wait_until_empty()

    def reset(self) -> None:
        self.streamer.reset()

    def close(self):
        self.player.close()
