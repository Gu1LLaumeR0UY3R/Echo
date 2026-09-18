# -*- coding: utf-8 -*-
"""
domain/app_entry.py
=====================
Entité pure du domaine -- aucune dépendance vers Windows, le registre, ou
quoi que ce soit de technique. Juste la donnée : "une application trouvée,
avec un nom affichable et un chemin vers son exécutable".
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AppEntry:
    name: str          # nom affichable (ex: "Google Chrome")
    exe_path: str       # chemin complet vers l'exécutable
    source: str          # d'où l'entrée vient : "registry", "folder_scan", "start_menu"
