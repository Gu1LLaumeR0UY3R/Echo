# -*- coding: utf-8 -*-
"""
main.py
========
Point d'entrée d'Echo. C'est le SEUL endroit du projet où les adapters
concrets (Qwen/Ollama, RVC streaming, Whisper) et l'interface graphique
sont assemblés ensemble -- partout ailleurs (app/, domain/, gui/), le code
ne connaît que les ports abstraits. C'est ce qui permet de changer de LLM
ou de moteur vocal plus tard sans toucher au reste.

Lancement :
    python main.py

-----------------------------------------------------------------------
MODIF VISUALISATION (ajout) :
`VisualizationRelay` est créé AVANT `RVCStreamingVoiceAdapter` (car il a
besoin d'un port de visualisation dès sa construction), puis passé à
`MainWindow` qui l'attache à son widget `WaveformVisualizer` une fois
celui-ci construit. Ça résout le problème d'ordre : l'audio existe avant
la fenêtre, mais le widget d'affichage a besoin de la fenêtre pour exister.
-----------------------------------------------------------------------
"""
from app.echo_service import EchoService
from adapters.ollama_qwen_adapter import OllamaQwenAdapter
from adapters.rvc_streaming_voice_adapter import RVCStreamingVoiceAdapter
from adapters.windows_app_launcher_adapter import WindowsAppLauncherAdapter
from gui.main_window import MainWindow


# ============================== CONFIG ====================================
HUBERT_PT_PATH = "models/hubert_base.pt"
RMVPE_PT_PATH = "models/rmvpe.pt"
VOICE_MODEL_PATH = "models/BT-7274.pth"     # ou "models/BT-7274.pth"
PITCH_SHIFT = 10                                     # 10 pour Miku, 0 pour BT-7274
EDGE_VOICE = "en-US-AriaNeural"

OLLAMA_MODEL = "qwen2.5:7b"      # vérifié : tient 100% en GPU à côté de RVC
SYSTEM_PROMPT = (
    "You are Miku, a friendly and lively voice assistant. Stay fully in "
    "character as Miku at all times -- never mention that you are Qwen, "
    "an AI model, or made by Alibaba Cloud. Keep every reply short: one "
    "or two short sentences maximum, conversational."
)
MAX_TOKENS = 60

# Micro : True active la reconnaissance vocale (nécessite faster-whisper,
# voir requirements.txt). False si vous voulez tester la GUI sans installer
# faster-whisper tout de suite -- le bouton micro sera juste désactivé.
ENABLE_MIC = True
WHISPER_MODEL_SIZE = "faster-whisper-base"   # "tiny" (plus rapide) / "base" / "small" (plus précis)
# ============================================================================


def main():
    print("[Echo] Chargement de la voix (HuBERT + RMVPE + Miku)...")
    voice = RVCStreamingVoiceAdapter(
        HUBERT_PT_PATH, RMVPE_PT_PATH, VOICE_MODEL_PATH,
        pitch_shift=PITCH_SHIFT, edge_voice=EDGE_VOICE, block_ms=300.0,
    )

    llm = OllamaQwenAdapter(model=OLLAMA_MODEL, system_prompt=SYSTEM_PROMPT, max_tokens=MAX_TOKENS)

    print("[Echo] Indexation des applications installées (registre + Menu Démarrer + dossiers)...")
    app_launcher = WindowsAppLauncherAdapter()

    speech = None
    if ENABLE_MIC:
        from adapters.vosk_speech_adapter import VoskSpeechAdapter
        speech = VoskSpeechAdapter(model_path="vosk_models/vosk-model-small-en-us-0.15", silence_duration_s=2.0,)

    echo_service = EchoService(llm, voice, app_launcher=app_launcher)

    print("[Echo] Prêt. Ouverture de la fenêtre...")
    app = MainWindow(echo_service, speech_adapter=speech, assistant_name="Echo")
    app.mainloop()


if __name__ == "__main__":
    main()
