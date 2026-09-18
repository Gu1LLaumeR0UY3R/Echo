# -*- coding: utf-8 -*-
"""
gui/waveform_visualizer.py
============================
Widget CustomTkinter : barres d'égaliseur qui montent depuis le bas,
réagissant à la voix de sortie d'Echo (TTS/RVC). Se redimensionne
automatiquement pour remplir son conteneur parent (pensé pour un layout
plein écran).

Implémente AudioVisualizationPort -- s'utilise directement comme
`visualizer=` passé à RVCStreamingVoiceAdapter (via VisualizationRelay),
sans rien connaître du pipeline audio de l'autre côté.

-----------------------------------------------------------------------
POURQUOI C'ÉTAIT SACCADÉ AVANT (et ce qui a changé) :
La version précédente recevait UN bloc audio toutes les ~300ms
(block_ms=300 côté RVCStreamingVoiceAdapter) et en tirait UNE seule image
par bloc -- donc l'affichage ne changeait que ~3 fois par seconde, ce qui
donne un effet haché même avec un timer d'animation à 30-35 fps.

Fix : chaque bloc reçu est maintenant découpé en plusieurs sous-morceaux
(SUBCHUNKS_PER_BLOCK, calculé pour correspondre à peu près à la durée
réelle du bloc divisée par l'intervalle d'une frame). Un seul sous-morceau
est consommé par frame d'animation -- l'affichage change donc à chaque
frame (~50 fps) au lieu de 3x/seconde, ce qui lisse le mouvement. En plus,
chaque barre interpole doucement (lerp) vers sa nouvelle cible plutôt que
de sauter directement dessus.
-----------------------------------------------------------------------

INTÉGRATION :
    self.waveform = WaveformVisualizer(parent_frame)
    self.waveform.grid(row=1, column=0, sticky="nsew")   # ou .pack(fill="both", expand=True)
"""
from __future__ import annotations

import queue
import numpy as np
import customtkinter as ctk

from ports.audio_visualization_port import AudioVisualizationPort


# Fréquence de rafraîchissement du canvas (~50 fps).
FRAME_INTERVAL_MS = 20
# Durée attendue d'un bloc audio, doit correspondre à `block_ms` passé à
# RVCStreamingVoiceAdapter côté main.py (300.0 par défaut). Sert à calculer
# combien de sous-morceaux découper par bloc pour un débit visuel régulier.
EXPECTED_BLOCK_MS = 300.0
SUBCHUNKS_PER_BLOCK = max(1, round(EXPECTED_BLOCK_MS / FRAME_INTERVAL_MS))

# Si aucun nouveau sous-morceau depuis ce délai, on considère qu'Echo a fini
# de parler et les barres retombent.
IDLE_TIMEOUT_MS = 200

BAR_COUNT = 48
BAR_GAP_RATIO = 0.35        # espace entre barres, proportionnel à leur largeur
MIN_BAR_HEIGHT_RATIO = 0.03  # hauteur minimale visible même à zéro (repos)
BAR_SMOOTHING = 0.35         # 0 < x <= 1 -- plus petit = mouvement plus doux/lent
AMPLITUDE_GAIN = 4.0         # gain visuel -- l'audio RMS est bas, on l'amplifie pour que ça "pop"

COLOR_BAR_LOW = "#1a3b42"    # bas de gamme (silence / faible amplitude)
COLOR_BAR_HIGH = "#00E5FF"   # haut de gamme (pic, cyan vif)


def _lerp_color(c1: str, c2: str, t: float) -> str:
    """Interpole deux couleurs hex ('#rrggbb') -- t dans [0, 1]."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


class WaveformVisualizer(ctk.CTkFrame, AudioVisualizationPort):
    def __init__(self, master, width: int = 800, height: int = 200, bar_count: int = BAR_COUNT, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self._width = width
        self._height = height
        self._bar_count = bar_count

        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._bar_heights = np.zeros(bar_count, dtype=np.float32)   # 0..1, valeur actuellement affichée
        self._ms_since_last_chunk = 0

        self.canvas = ctk.CTkCanvas(
            self,
            width=width,
            height=height,
            highlightthickness=0,
            bg=self._resolve_bg_color(),
        )
        self.canvas.pack(fill="both", expand=True)
        # Se redimensionne avec son conteneur -- essentiel pour le layout plein écran.
        self.canvas.bind("<Configure>", self._on_resize)

        self._draw_bars()
        self._animate()

    # ------------------------------------------------------------------
    # AudioVisualizationPort
    # ------------------------------------------------------------------
    def push_samples(self, samples: np.ndarray) -> None:
        """
        Appelé depuis le thread audio/RVC. Non-bloquant : on découpe le
        bloc en sous-morceaux et on les empile -- AUCUN accès Tkinter ici.
        """
        if samples is None or samples.size == 0:
            return
        subchunks = np.array_split(samples, SUBCHUNKS_PER_BLOCK)
        for chunk in subchunks:
            if chunk.size:
                self._queue.put(chunk)
        # Garde-fou : si jamais la queue s'accumule (frame drop, lag GUI),
        # on jette les plus vieux morceaux plutôt que de prendre du retard
        # indéfiniment sur l'audio réel.
        while self._queue.qsize() > SUBCHUNKS_PER_BLOCK * 4:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Redimensionnement
    # ------------------------------------------------------------------
    def _on_resize(self, event):
        self._width = event.width
        self._height = event.height

    # ------------------------------------------------------------------
    # Rendu (thread principal Tkinter uniquement)
    # ------------------------------------------------------------------
    def _animate(self):
        chunk = self._next_chunk()

        if chunk is not None:
            self._ms_since_last_chunk = 0
            target = self._compute_bar_amplitudes(chunk)
        else:
            self._ms_since_last_chunk += FRAME_INTERVAL_MS
            target = np.zeros(self._bar_count, dtype=np.float32) if self._ms_since_last_chunk >= IDLE_TIMEOUT_MS else None

        if target is not None:
            self._bar_heights += (target - self._bar_heights) * BAR_SMOOTHING
            self._draw_bars()

        self.after(FRAME_INTERVAL_MS, self._animate)

    def _next_chunk(self) -> "np.ndarray | None":
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _compute_bar_amplitudes(self, chunk: np.ndarray) -> np.ndarray:
        """RMS par segment, un segment par barre, amplifié et clippé à [0, 1]."""
        segments = np.array_split(chunk, self._bar_count)
        amplitudes = np.empty(self._bar_count, dtype=np.float32)
        for i, seg in enumerate(segments):
            if seg.size == 0:
                amplitudes[i] = 0.0
                continue
            rms = float(np.sqrt(np.mean(np.square(seg))))
            amplitudes[i] = min(1.0, rms * AMPLITUDE_GAIN)
        return amplitudes

    def _draw_bars(self):
        w, h = self._width, self._height
        if w <= 1 or h <= 1:
            return

        n = self._bar_count
        slot_w = w / n
        bar_w = slot_w * (1.0 - BAR_GAP_RATIO)
        gap = (slot_w - bar_w) / 2
        min_h = h * MIN_BAR_HEIGHT_RATIO

        self.canvas.delete("bar")
        for i in range(n):
            level = float(self._bar_heights[i])
            bar_h = max(min_h, level * h * 0.95)
            x0 = i * slot_w + gap
            x1 = x0 + bar_w
            y1 = h
            y0 = h - bar_h
            color = _lerp_color(COLOR_BAR_LOW, COLOR_BAR_HIGH, level)
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="", tags="bar")

    def _resolve_bg_color(self) -> str:
        """
        CTkCanvas (tk.Canvas pur) n'a pas de fg_color CTk -- on récupère la
        couleur de fond du thème courant pour que le canvas se fonde dans
        l'interface au lieu d'avoir un rectangle blanc/gris qui détonne.
        """
        try:
            mode = ctk.get_appearance_mode()  # "Light" ou "Dark"
            theme_bg = ctk.ThemeManager.theme["CTkFrame"]["fg_color"]
            return theme_bg[1] if mode == "Dark" else theme_bg[0]
        except Exception:
            return "#1a1a1a"
