# -*- coding: utf-8 -*-
"""
ports/text_generation_port.py
==============================
Contrat pour "générer une réponse texte à partir d'un historique de
conversation". Peu importe QUI répond derrière (Qwen aujourd'hui, un autre
modèle demain) -- le reste de l'application (app/echo_service.py, gui/)
ne dépend que de CE contrat, jamais de Qwen/Ollama directement.
"""
from abc import ABC, abstractmethod
from typing import List

from domain.message import Message


class TextGenerationPort(ABC):
    @abstractmethod
    def generate(self, history: List[Message]) -> str:
        """
        Prend l'historique de conversation (le dernier message étant celui
        de l'utilisateur auquel il faut répondre) et retourne la réponse
        texte générée. Le "system prompt" (personnalité, contraintes de
        longueur, etc.) est un détail d'implémentation de l'adapter, pas
        une préoccupation de ce contrat.
        """
        raise NotImplementedError
