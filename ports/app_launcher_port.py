# -*- coding: utf-8 -*-
"""
ports/app_launcher_port.py
=============================
Contrat pour "trouver une application installée sur le PC, et l'ouvrir".
L'implémentation concrète (scan registre Windows + dossiers + raccourcis
Menu Démarrer) vit dans adapters/windows_app_launcher_adapter.py -- le
reste de l'app (app/, gui/) ne connaît que ce port.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from domain.app_entry import AppEntry


class AppLauncherPort(ABC):
    @abstractmethod
    def find(self, query: str) -> Optional[AppEntry]:
        """
        Cherche la meilleure correspondance pour `query` dans l'index des
        applications installées. Essaie d'abord une correspondance
        exacte/substring, puis une recherche floue si rien d'exact n'est
        trouvé. Retourne None si rien d'assez proche n'est trouvé.
        """
        raise NotImplementedError

    @abstractmethod
    def launch(self, entry: AppEntry) -> None:
        """Lance l'exécutable désigné par `entry`."""
        raise NotImplementedError

    @abstractmethod
    def refresh_index(self) -> int:
        """
        Relance un scan complet (registre + dossiers + Menu Démarrer) et
        reconstruit l'index. Retourne le nombre d'applications trouvées.
        À appeler au premier lancement (pas de cache) ou sur demande
        explicite de l'utilisateur ("rafraîchis tes applications").
        """
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> List[AppEntry]:
        """Retourne toutes les applications actuellement indexées."""
        raise NotImplementedError
