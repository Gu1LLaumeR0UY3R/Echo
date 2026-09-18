# -*- coding: utf-8 -*-
"""
gui/main_window.py
====================
Interface graphique d'Echo (CustomTkinter -- rendu moderne, sombre par
défaut, une seule dépendance `pip install customtkinter`).

Point technique important : toute la logique LLM+voix (potentiellement
plusieurs secondes) tourne dans un THREAD SÉPARÉ, jamais dans le thread
principal Tkinter -- sinon la fenêtre se figerait (pas de rafraîchissement
possible) pendant que Qwen réfléchit et que Miku parle. Les mises à jour
de l'interface depuis ce thread d'arrière-plan passent TOUJOURS par
`self.after(0, ...)`, qui est la seule façon thread-safe de toucher aux
widgets Tkinter depuis un autre thread que le sien.

-----------------------------------------------------------------------
RETOUR EN ARRIÈRE (ajout) :
Fenêtre classique redimensionnable (plus de plein écran forcé), et
waveform retirée (le point d'accroche audio n'était pas synchro avec la
lecture réelle -- voir la conversation précédente). Le widget
`WaveformVisualizer` et `VisualizationRelay` restent dans le projet, non
utilisés, pour le jour où le point d'accroche temps réel sera revu.
-----------------------------------------------------------------------
"""
import threading

import customtkinter as ctk

from app.echo_service import EchoService


class MainWindow(ctk.CTk):
    def __init__(self, echo_service: EchoService, speech_adapter=None, assistant_name: str = "Echo"):
        super().__init__()
        self.echo_service = echo_service
        self.speech_adapter = speech_adapter
        self.assistant_name = assistant_name
        self._is_recording = False

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Echo")
        self.geometry("560x680")
        self.minsize(420, 480)
        self.resizable(True, True)

        # --- Journal de conversation ---
        self.chat_log = ctk.CTkTextbox(self, wrap="word", state="disabled", font=("Segoe UI", 13))
        self.chat_log.pack(padx=12, pady=(12, 6), fill="both", expand=True)

        # --- Statut ---
        self.status_label = ctk.CTkLabel(self, text="Prêt.", text_color="gray60", anchor="w")
        self.status_label.pack(padx=12, fill="x")

        # --- Barre de saisie ---
        input_row = ctk.CTkFrame(self, fg_color="transparent")
        input_row.pack(padx=12, pady=12, fill="x")

        self.entry = ctk.CTkEntry(input_row, placeholder_text="Écrivez à Echo...")
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.entry.bind("<Return>", lambda _e: self._on_send())

        self.send_button = ctk.CTkButton(input_row, text="Envoyer", width=90, command=self._on_send)
        self.send_button.pack(side="left", padx=(0, 8))

        self.mic_button = ctk.CTkButton(
            input_row, text="🎙", width=48, fg_color="gray30", command=self._on_mic_toggle
        )
        self.mic_button.pack(side="left")
        if self.speech_adapter is None:
            self.mic_button.configure(state="disabled")

        # Aperçu en direct de ce que dit l'utilisateur pendant l'enregistrement
        # (rempli par les callbacks on_partial de l'adapter micro). Vide/masqué
        # hors enregistrement.
        self.live_preview_label = ctk.CTkLabel(
            self, text="", text_color="#5dade2", anchor="w", font=("Segoe UI", 12, "italic")
        )
        self.live_preview_label.pack(padx=12, pady=(0, 6), fill="x")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- UI --
    def _append_message(self, speaker: str, text: str):
        self.chat_log.configure(state="normal")
        self.chat_log.insert("end", f"{speaker} : {text}\n\n")
        self.chat_log.configure(state="disabled")
        self.chat_log.see("end")

    def _set_status(self, text: str):
        self.status_label.configure(text=text)

    def _set_ui_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.entry.configure(state=state)
        self.send_button.configure(state=state)
        if self.speech_adapter is not None:
            self.mic_button.configure(state="disabled" if busy else "normal")

    # ----------------------------------------------------- Envoi texte --
    def _on_send(self):
        user_text = self.entry.get().strip()
        if not user_text:
            return
        self.entry.delete(0, "end")
        self._append_message("Vous", user_text)
        self._set_ui_busy(True)
        threading.Thread(target=self._process_in_background, args=(user_text,), daemon=True).start()

    def _process_in_background(self, user_text: str):
        def status_cb(stage: str, text):
            label = {
                "thinking": "Qwen réfléchit...",
                "launching": f"Ouverture de « {text} »..." if text else "Ouverture...",
                "speaking": f"{self.assistant_name} parle...",
                "done": "Prêt.",
            }.get(stage, "")
            self.after(0, lambda: self._set_status(label))
            if stage == "speaking" and text:
                self.after(0, lambda: self._append_message(self.assistant_name, text))

        try:
            self.echo_service.handle_user_text(user_text, on_status=status_cb)
        except Exception as exc:  # noqa: BLE001 -- on affiche l'erreur plutôt que de planter la GUI
            self.after(0, lambda: self._append_message("Erreur", str(exc)))
            self.after(0, lambda: self._set_status("Erreur -- voir le journal."))
        finally:
            self.after(0, lambda: self._set_ui_busy(False))

    # ------------------------------------------------------- Micro --
    def _on_mic_toggle(self):
        if not self._is_recording:
            self._start_recording()
        else:
            # Clic manuel pendant l'enregistrement = arrêt forcé, au cas où
            # l'arrêt automatique sur silence tarderait ou ne se déclenche
            # pas (bruit de fond, micro trop sensible, etc.).
            self._finish_recording()

    def _start_recording(self):
        self._is_recording = True
        self.mic_button.configure(fg_color="#c0392b", text="⏹")
        self._set_status("Je vous écoute...")
        self.live_preview_label.configure(text="")

        def on_partial(text: str):
            self.after(0, lambda: self.live_preview_label.configure(text=f"« {text} »"))

        def on_silence_stop():
            # Appelé depuis le thread audio (callback micro) -- on ne fait
            # que déclencher la suite depuis le thread principal, jamais de
            # traitement lourd ici.
            self.after(0, self._finish_recording)

        self.speech_adapter.start_recording(on_partial=on_partial, on_silence_stop=on_silence_stop)

    def _finish_recording(self):
        if not self._is_recording:
            return  # déjà arrêté (évite un double déclenchement silence + clic manuel)
        self._is_recording = False
        self.mic_button.configure(fg_color="gray30", text="🎙")
        self.live_preview_label.configure(text="")
        self._set_status("Transcription en cours...")
        self._set_ui_busy(True)
        threading.Thread(target=self._stop_and_process_mic, daemon=True).start()

    def _stop_and_process_mic(self):
        text = self.speech_adapter.stop_recording_and_transcribe()
        if not text:
            self.after(0, lambda: self._set_status("Rien compris -- réessayez."))
            self.after(0, lambda: self._set_ui_busy(False))
            return
        self.after(0, lambda: self._append_message("Vous (voix)", text))
        self._process_in_background(text)

    # ------------------------------------------------------- Fermeture --
    def _on_close(self):
        try:
            self.echo_service.voice.close()
        except Exception:
            pass
        self.destroy()
