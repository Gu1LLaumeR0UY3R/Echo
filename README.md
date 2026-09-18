# Echo — Architecture hexagonale + interface graphique

## Arborescence

```
Echo/
├── domain/
│   └── message.py                    <- Message (role, content, timestamp). Zéro dépendance externe.
├── ports/                             <- Les CONTRATS (interfaces abstraites)
│   ├── text_generation_port.py        (generate(history) -> str)
│   ├── voice_output_port.py           (speak(text), reset())
│   └── speech_input_port.py           (start_recording(), stop_recording_and_transcribe())
├── adapters/                          <- Les IMPLÉMENTATIONS concrètes
│   ├── ollama_qwen_adapter.py          Qwen via Ollama, implémente TextGenerationPort
│   ├── rvc_streaming_voice_adapter.py  Miku/BT-7274 en streaming, implémente VoiceOutputPort
│   └── whisper_speech_adapter.py       faster-whisper, implémente SpeechInputPort
├── app/
│   └── echo_service.py                Orchestration : historique + enchaînement LLM -> voix
├── gui/
│   └── main_window.py                 Fenêtre CustomTkinter (chat + bouton micro)
├── main.py                            Point d'entrée : assemble tout, lance la fenêtre
│
├── engine.py, pipeline.py, hubert_engine.py, rmvpe_engine.py, arch/   <- DÉJÀ CHEZ VOUS
├── models/                                                             <- DÉJÀ CHEZ VOUS (vos .pth/.pt)
├── stream_config.py, sola_streamer.py, infer_adapter.py,
│   streaming_config_rvc.py, tts_streaming_bridge.py                    <- Couche streaming déjà validée
│
└── requirements.txt
```

## Pourquoi cette organisation (le principe de l'architecture hexagonale)

- **`domain/`** ne dépend de RIEN d'externe (pas de Qwen, pas de RVC, pas de Tkinter). C'est la définition la plus neutre de "c'est quoi un message".
- **`ports/`** définit des CONTRATS abstraits : "quelque chose qui sait générer du texte", "quelque chose qui sait parler". Ni `app/` ni `gui/` ne savent que c'est Qwen ou Miku derrière.
- **`adapters/`** implémente ces contrats avec les technologies concrètes actuelles.
- **`app/echo_service.py`** orchestre le tout via les ports uniquement.
- **`main.py`** est le SEUL endroit qui choisit "avec quels adapters concrets on assemble tout" — c'est le point de changement si vous voulez un jour remplacer Qwen par un autre LLM, ou Miku par BT-7274 par défaut : une ligne à changer dans `main.py`, rien d'autre.

## Installation

```powershell
cd Echo
pip install -r requirements.txt
ffmpeg -version   # doit fonctionner (voir mise en place précédente)
```

Si `torch`/`torchaudio` ne sont pas déjà en version CUDA dans votre venv :
```powershell
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## Avant de lancer

Vérifiez dans `main.py` (section `CONFIG` en haut) :
- Les 3 chemins (`HUBERT_PT_PATH`, `RMVPE_PT_PATH`, `VOICE_MODEL_PATH`) — normalement déjà bons si `models/` contient vos fichiers comme avant
- `OLLAMA_MODEL = "qwen2.5:7b"` — vérifiez avec `ollama list`
- `PITCH_SHIFT` — 10 pour Miku, 0 pour BT-7274

## Lancement

```powershell
python main.py
```

Une fenêtre s'ouvre : zone de conversation en haut, ligne de statut, puis
zone de saisie + bouton "Envoyer" + bouton micro 🎙 en bas.

- **Texte** : tapez, Entrée ou "Envoyer". Le statut affiche "Qwen réfléchit..." puis "Miku parle...", puis "Prêt."
- **Micro** : cliquez 🎙 (devient rouge ⏹, enregistrement en cours), reparlez, recliquez pour arrêter. Le texte transcrit apparaît dans le journal, puis suit le même chemin que si vous l'aviez tapé.

## Ce qui a été testé (dans ce conteneur, sans vos vrais modèles/Ollama)

- `domain/` + `ports/` + `app/echo_service.py` : testé avec de faux adapters — historique de conversation, ordre des callbacks de statut (thinking → speaking → done), appels voix/reset au bon moment. **Fonctionne comme prévu.**
- `gui/main_window.py` : rendu réel de la fenêtre CustomTkinter testé (headless), plus un envoi de message complet simulé avec un vrai `mainloop()` en cours d'exécution (thread d'arrière-plan → mise à jour du journal → statut). **Fonctionne comme prévu.**
- `adapters/ollama_qwen_adapter.py`, `adapters/rvc_streaming_voice_adapter.py`, `adapters/whisper_speech_adapter.py` : syntaxe vérifiée, mais **pas testés en conditions réelles** ici (pas d'Ollama, pas de GPU, pas de micro dans ce conteneur). Ce sont les mêmes briques que TEST 3/4/5/6 déjà validées chez vous pour la partie RVC/Qwen — seul `whisper_speech_adapter.py` est entièrement nouveau et n'a encore jamais tourné.

## Si quelque chose ne marche pas

- `ModuleNotFoundError: No module named 'customtkinter'` → `pip install customtkinter`
- `ModuleNotFoundError: No module named 'faster_whisper'` → `pip install faster-whisper`, ou passez `ENABLE_MIC = False` dans `main.py` si vous voulez d'abord tester sans le micro
- Micro : premier lancement avec `WHISPER_MODEL_SIZE = "base"` télécharge le modèle (~150MB, une seule fois, mis en cache)
- Le clic micro ne fait rien / erreur immédiate → vérifiez qu'un micro est bien détecté par Windows (Paramètres son)
- Texte transcrit vide après avoir parlé → parlez plus fort/plus longtemps (moins de 300ms d'enregistrement est ignoré volontairement, voir `whisper_speech_adapter.py`)
