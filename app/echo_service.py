# -*- coding: utf-8 -*-
"""
app/echo_service.py
=====================
Le "chef d'orchestre" applicatif : ne connaît AUCUN détail de Qwen, RVC ou
Tkinter -- seulement les ports (TextGenerationPort, VoiceOutputPort,
AppLauncherPort). Ça veut dire qu'on pourrait remplacer Qwen par un autre
LLM, ou Miku par un autre moteur vocal, sans toucher une ligne de ce
fichier -- il suffit d'écrire un nouvel adapter qui respecte le même port.

-----------------------------------------------------------------------
MODIF OUVERTURE D'APPLICATIONS (ajout) :
Avant d'envoyer le texte au LLM, on teste s'il commence par un verbe
déclencheur ("ouvre", "lance", "démarre", "open", "run", "start"). Si
oui, c'est traité comme une COMMANDE directe, PAS une question de
conversation :
    1. Le reste de la phrase est cherché dans l'index des applications
       (AppLauncherPort.find -- exact/substring d'abord, flou en repli)
    2. Si trouvée -> lancement direct + courte confirmation vocale
    3. Si pas trouvée -> message d'échec, sans jamais appeler le LLM

C'est un chemin RAPIDE et déterministe qui court-circuite Qwen pour ce
cas précis (pas besoin d'attendre 5-9s de génération pour "ouvre Chrome").
Si aucun verbe déclencheur n'est détecté, ou si `app_launcher` n'a pas été
fourni, le comportement est identique à avant (conversation normale).
-----------------------------------------------------------------------
"""
import re
from typing import Callable, List, Optional

from domain.message import Message
from ports.text_generation_port import TextGenerationPort
from ports.voice_output_port import VoiceOutputPort
from ports.app_launcher_port import AppLauncherPort


# Callback optionnel appelé à chaque étape, pour que l'interface graphique
# puisse afficher "Qwen réfléchit...", "Miku parle...", etc. sans que ce
# fichier ait besoin de connaître Tkinter.
#   on_status(stage: str, text: Optional[str])
#   stage prend les valeurs : "thinking", "launching", "speaking", "done"
#   - "launching" : text = la requête d'appli extraite (ex. "chrome")
#   - "speaking"  : text = la réponse générée / le message de confirmation
StatusCallback = Callable[[str, Optional[str]], None]

# Verbes déclencheurs, FR + EN. Capture tout ce qui suit comme requête
# d'application. Insensible à la casse, ancré en début de phrase.
_LAUNCH_PATTERN = re.compile(
    r"^(?:ouvre|lance|d[ée]marre|d[ée]marrer|open|run|start)\s+(.+)$",
    re.IGNORECASE,
)


class EchoService:
    def __init__(
        self,
        llm: TextGenerationPort,
        voice: VoiceOutputPort,
        app_launcher: Optional[AppLauncherPort] = None,
    ):
        self.llm = llm
        self.voice = voice
        self.app_launcher = app_launcher
        self.history: List[Message] = []

    def handle_user_text(self, user_text: str, on_status: Optional[StatusCallback] = None) -> str:
        """
        Point d'entrée principal : que le texte vienne du clavier ou d'une
        transcription vocale (Whisper), le traitement est identique à
        partir d'ici.
        """
        app_query = self._extract_app_launch_query(user_text)
        if app_query and self.app_launcher is not None:
            return self._handle_app_launch(user_text, app_query, on_status)

        self.history.append(Message(role="user", content=user_text))

        if on_status:
            on_status("thinking", None)
        reply = self.llm.generate(self.history)
        self.history.append(Message(role="assistant", content=reply))

        if on_status:
            on_status("speaking", reply)
        self.voice.reset()
        self.voice.speak(reply)

        if on_status:
            on_status("done", None)
        return reply

    def reset_conversation(self):
        """Efface l'historique -- utile pour repartir sur une nouvelle discussion."""
        self.history = []
        self.voice.reset()

    # ------------------------------------------------------------------
    # Ouverture d'applications
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_app_launch_query(user_text: str) -> Optional[str]:
        match = _LAUNCH_PATTERN.match(user_text.strip())
        return match.group(1).strip() if match else None

    def _handle_app_launch(
        self, original_text: str, query: str, on_status: Optional[StatusCallback]
    ) -> str:
        if on_status:
            on_status("launching", query)

        entry = self.app_launcher.find(query)
        if entry is None:
            reply = f"Je ne trouve pas d'application correspondant à « {query} »."
        else:
            try:
                self.app_launcher.launch(entry)
                reply = f"J'ouvre {entry.name}."
            except Exception as exc:  # noqa: BLE001 -- on renvoie l'erreur en confirmation vocale plutôt que de planter
                reply = f"Je n'ai pas réussi à ouvrir {entry.name} : {exc}"

        # On garde une trace dans l'historique pour que le LLM ait le
        # contexte si la conversation continue juste après ("et ferme-le
        # dans 5 minutes" etc.), même si le LLM n'a pas généré ce tour-ci.
        self.history.append(Message(role="user", content=original_text))
        self.history.append(Message(role="assistant", content=reply))

        if on_status:
            on_status("speaking", reply)
        self.voice.reset()
        self.voice.speak(reply)

        if on_status:
            on_status("done", None)
        return reply
