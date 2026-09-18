# -*- coding: utf-8 -*-
"""
tts_streaming_bridge.py
========================
Le pipeline complet : texte -> Edge-TTS (streaming) -> RVC (streaming,
SOLAStreamer) -> haut-parleurs, sans attendre la phrase complète avant de
commencer à parler.

POINT TECHNIQUE IMPORTANT (vérifié sur le code source d'edge-tts) :
la librairie Python `edge-tts` streame le résultat au format MP3
("audio-24khz-48kbitrate-mono-mp3"), codé en dur côté serveur Microsoft —
il n'y a pas d'option pour demander du PCM brut directement en streaming
comme il y en a côté Node.js. On doit donc décoder ce flux MP3 au fur et à
mesure qu'il arrive, PAS attendre le fichier complet.

Solution : on pipe les octets MP3 vers un process `ffmpeg` en sous-process
au fur et à mesure qu'ils arrivent (stdin), et on lit le PCM décodé en
continu depuis sa sortie (stdout). ffmpeg sait décoder du MP3 de façon
progressive, sans avoir besoin du fichier entier — c'est le même principe
que le streaming radio.

PRÉREQUIS : ffmpeg doit être installé et accessible dans le PATH Windows
(commande `ffmpeg -version` doit fonctionner dans un terminal). Si Edge-TTS
fonctionnait déjà chez vous en V1, il y a de bonnes chances qu'il soit déjà
là — sinon : https://www.gyan.dev/ffmpeg/builds/ (build "essentials",
ajouter le dossier bin/ au PATH).

-----------------------------------------------------------------------
MODIF VISUALISATION (ajout) :
`RealtimePlayer` accepte maintenant un callback optionnel `on_block`,
appelé à chaque bloc audio poussé vers la sortie -- AVANT qu'il joue,
donc la visualisation est synchro avec ce qui va sortir des enceintes.
Le callback est appelé depuis le thread qui fait la conversion RVC (pas
le thread audio temps réel de sounddevice), donc il ne doit pas être
lourd -- il doit juste empiler dans une queue et retourner immédiatement.
Si `on_block` est None (comportement par défaut), rien ne change par
rapport à avant.
-----------------------------------------------------------------------
"""
from __future__ import annotations

import asyncio
import queue
import subprocess
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

try:
    import edge_tts
except ImportError as e:
    raise ImportError(
        "La librairie edge-tts n'est pas installée. `pip install edge-tts`."
    ) from e

from sola_streamer import SOLAStreamer, AudioBlockizer


class FFmpegMp3StreamDecoder:
    """
    Décodeur MP3 progressif basé sur ffmpeg en sous-process.
    On lui envoie des octets MP3 au fil de l'eau (feed), et on récupère du
    PCM float32 mono au taux d'échantillonnage voulu (read_available).
    """

    def __init__(self, target_sample_rate: int):
        self.target_sample_rate = target_sample_rate
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-f", "mp3", "-i", "pipe:0",
                "-f", "f32le",              # PCM float32 little-endian brut
                "-ar", str(target_sample_rate),
                "-ac", "1",                  # mono
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._out_queue: "queue.Queue[bytes]" = queue.Queue()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self):
        # Lit stdout par petits paquets dès qu'ils sont disponibles, plutôt
        # qu'attendre un gros buffer -- c'est ça qui permet au son de sortir
        # progressivement au lieu d'un seul bloc à la fin.
        while True:
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                break
            self._out_queue.put(chunk)

    def feed(self, mp3_bytes: bytes):
        """Envoie de nouveaux octets MP3 reçus d'Edge-TTS."""
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.write(mp3_bytes)
            self._proc.stdin.flush()

    def read_available(self) -> np.ndarray:
        """Récupère tout le PCM décodé disponible pour l'instant (peut être vide)."""
        chunks = []
        try:
            while True:
                chunks.append(self._out_queue.get_nowait())
        except queue.Empty:
            pass
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        raw = b"".join(chunks)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def close_input(self):
        """Signale la fin de l'envoi (plus de texte à venir pour cette phrase)."""
        if self._proc.stdin and not self._proc.stdin.closed:
            try:
                self._proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

    def drain_remaining(self, timeout: float = 5.0) -> np.ndarray:
        """
        Attend la fin du décodage et retourne tout le PCM restant.
        À appeler après close_input(), une fois qu'on n'enverra plus rien.
        """
        self._reader_thread.join(timeout=timeout)
        return self.read_available()

    def terminate(self):
        try:
            self._proc.kill()
        except Exception:
            pass


class RealtimePlayer:
    """
    Sortie audio continue : on pousse des blocs convertis dans une file, et
    un flux sounddevice les joue dès qu'ils sont disponibles, sans bloquer
    le thread qui fait la conversion RVC pendant ce temps.

    `on_block` (optionnel) : callback(np.ndarray) appelé à chaque `push()`,
    pour brancher une visualisation temps réel sans coupler ce fichier à
    la GUI. Toute exception levée dedans est capturée et ignorée -- ça ne
    doit jamais casser l'audio.
    """

    def __init__(
        self,
        sample_rate: int,
        block_frame: int,
        on_block: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.block_frame = block_frame
        self.on_block = on_block
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            blocksize=block_frame,
            channels=1,
            dtype="float32",
            latency="high",
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print(status)
        try:
            block = self._queue.get_nowait()
        except queue.Empty:
            block = np.zeros(frames, dtype=np.float32)
        if len(block) < frames:
            block = np.pad(block, (0, frames - len(block)))
        outdata[:, 0] = block[:frames]

    def push(self, block: np.ndarray):
        self._queue.put(block)
        if self.on_block is not None:
            try:
                self.on_block(block)
            except Exception as e:
                # Une erreur de visualisation ne doit jamais couper le son.
                print(f"[RealtimePlayer] on_block a levé une exception (ignorée) : {e}")

    def wait_until_empty(self, poll_s: float = 0.05):
        import time
        while not self._queue.empty():
            time.sleep(poll_s)
        # petite marge pour laisser le dernier bloc finir de jouer
        import time as _t
        _t.sleep(self.block_frame / self.sample_rate + 0.1)

    def close(self):
        self._stream.stop()
        self._stream.close()


async def speak_streaming(
    text: str,
    streamer: SOLAStreamer,
    cfg,
    voice: str = "en-US-AriaNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz",
    player: "RealtimePlayer | None" = None,
):
    """
    Point d'entrée principal : envoie `text` à Edge-TTS, décode le MP3 au
    fil de l'eau, convertit avec `streamer` (déjà configuré avec votre
    VoiceEngine via infer_adapter.make_lite_engine_infer_fn), et joue le
    résultat en continu.

    Réutilise le SOLAStreamer/cfg déjà construits ailleurs dans votre
    programme -- ne recrée rien, pour garder le contexte (sola_buffer,
    input_wav) cohérent d'un appel à l'autre si vous enchaînez plusieurs
    phrases avec la même instance de streamer.
    """
    own_player = player is None
    if player is None:
        player = RealtimePlayer(cfg.sample_rate, cfg.block_frame)

    decoder = FFmpegMp3StreamDecoder(cfg.sample_rate)
    blockizer = AudioBlockizer(cfg.block_frame)

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)

    # Tampon de sécurité : on garde les tout premiers blocs convertis en
    # mémoire (pas encore envoyés à la sortie audio) le temps d'en avoir
    # PREBUFFER_BLOCKS d'avance. Ça absorbe les petits à-coups de timing en
    # tout début de phrase (réseau Edge-TTS, décodage MP3, premier calcul
    # RVC) qui provoquaient une micro-coupure au début du son.
    PREBUFFER_BLOCKS = 3
    pending_blocks = []

    def _push(block):
        if len(pending_blocks) < PREBUFFER_BLOCKS:
            pending_blocks.append(block)
            if len(pending_blocks) == PREBUFFER_BLOCKS:
                for b in pending_blocks:
                    player.push(b)
                pending_blocks.clear()
        else:
            player.push(block)

    try:
        async for chunk in communicate.stream():
            if chunk["type"] != "audio":
                continue
            decoder.feed(chunk["data"])
            pcm = decoder.read_available()
            if pcm.size == 0:
                continue
            for block in blockizer.feed(pcm):
                converted = streamer.push_block(block)
                _push(converted)

        # Fin du texte : on vide ce qu'il reste à décoder/convertir.
        decoder.close_input()
        pcm = decoder.drain_remaining()
        if pcm.size:
            for block in blockizer.feed(pcm):
                converted = streamer.push_block(block)
                _push(converted)
        for block in blockizer.flush():
            converted = streamer.push_block(block)
            _push(converted)

        # Si la phrase était très courte (moins de PREBUFFER_BLOCKS blocs
        # au total), le tampon n'a jamais été vidé automatiquement -- on
        # le vide ici pour ne pas perdre la fin de la phrase.
        for b in pending_blocks:
            player.push(b)
        pending_blocks.clear()

        if own_player:
            player.wait_until_empty()
    finally:
        decoder.terminate()
        if own_player:
            player.close()
