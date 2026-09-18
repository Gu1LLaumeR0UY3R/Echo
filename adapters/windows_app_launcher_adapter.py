# -*- coding: utf-8 -*-
"""
adapters/windows_app_launcher_adapter.py
===========================================
Implémentation concrète de AppLauncherPort pour Windows.

SOURCES D'INDEXATION (dans cet ordre de priorité pour le nom affiché,
mais toutes fusionnées dans le même index) :
  1. REGISTRE -- HKLM/HKCU \\Software\\Microsoft\\Windows\\CurrentVersion\\
     Uninstall\\*, plus l'équivalent Wow6432Node (apps 32-bit sur Windows
     64-bit). Chaque appli installée "proprement" a une clé avec un
     DisplayName et souvent un DisplayIcon ou InstallLocation pointant
     vers l'exécutable principal.
  2. MENU DÉMARRER -- les raccourcis .lnk dans les dossiers Programs
     (utilisateur + tous les utilisateurs), résolus vers leur cible réelle
     via pywin32 (COM WScript.Shell). C'est souvent la source la plus
     fiable pour le nom "humain" d'une appli.
  3. DOSSIERS STANDARDS -- scan récursif limité en profondeur de
     Program Files, Program Files (x86), %LOCALAPPDATA%\\Programs, en
     filet de sécurité pour les .exe non couverts par les deux sources
     au-dessus.

PRÉREQUIS : `pip install pywin32` (pour résoudre les .lnk). Si pywin32
n'est pas installé, l'adapter fonctionne quand même -- il saute juste la
source Menu Démarrer et log un avertissement.

CACHE : l'index complet est sauvegardé en JSON (cache_path) pour éviter
de rescanner registre+disque à chaque lancement d'Echo (le scan complet
peut prendre plusieurs secondes). Le cache est chargé automatiquement à
la construction s'il existe ; sinon, un refresh_index() complet est
déclenché une première fois.
"""
from __future__ import annotations

import difflib
import json
import os
import subprocess
import winreg
from pathlib import Path
from typing import List, Optional

from domain.app_entry import AppEntry
from ports.app_launcher_port import AppLauncherPort

try:
    import win32com.client  # pywin32 -- pour résoudre les raccourcis .lnk
    _HAS_PYWIN32 = True
except ImportError:
    _HAS_PYWIN32 = False


# Dossiers standards scannés en filet de sécurité (profondeur limitée pour
# rester rapide -- pas un scan de tout le disque).
_STANDARD_FOLDERS = [
    os.environ.get("ProgramFiles", r"C:\Program Files"),
    os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
]
_FOLDER_SCAN_MAX_DEPTH = 3

_START_MENU_FOLDERS = [
    os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs"),
    r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs",
]

_UNINSTALL_REGISTRY_ROOTS = [
    (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
    (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
]

_FUZZY_MATCH_CUTOFF = 0.6


class WindowsAppLauncherAdapter(AppLauncherPort):
    def __init__(self, cache_path: str = "app_index_cache.json"):
        self.cache_path = cache_path
        self._index: List[AppEntry] = []

        if os.path.exists(self.cache_path):
            self._load_cache()
        else:
            print("[AppLauncher] Pas de cache trouvé, indexation initiale...")
            self.refresh_index()

    # ------------------------------------------------------------------
    # AppLauncherPort
    # ------------------------------------------------------------------
    def find(self, query: str) -> Optional[AppEntry]:
        if not query or not self._index:
            return None
        q = query.strip().lower()

        # 1) Correspondance exacte ou substring (rapide, prioritaire).
        for entry in self._index:
            name_l = entry.name.lower()
            if q == name_l or q in name_l or name_l in q:
                return entry

        # 2) Recherche floue en fallback (fautes de frappe, formulation
        #    approximative captée par le LLM -- ex. "bloc-notes" ne
        #    matchera pas "notepad" par substring mais pourrait matcher
        #    via une faute de frappe sur le nom réel).
        names = [e.name for e in self._index]
        matches = difflib.get_close_matches(query, names, n=1, cutoff=_FUZZY_MATCH_CUTOFF)
        if matches:
            best_name = matches[0]
            for entry in self._index:
                if entry.name == best_name:
                    return entry
        return None

    def launch(self, entry: AppEntry) -> None:
        if not os.path.exists(entry.exe_path):
            raise FileNotFoundError(f"Exécutable introuvable : {entry.exe_path}")

        real_path = self._resolve_real_executable(entry)
        print(f"[AppLauncher] Lancement : {entry.name} -> {real_path}")

        # os.startfile est l'équivalent d'un double-clic Windows -- gère
        # correctement les .exe qui ont besoin de leur dossier de travail,
        # les élévations UAC déjà configurées, etc. Plus fiable que
        # subprocess.Popen pour des applis tierces variées.
        os.startfile(real_path)  # nosec -- lancement volontaire d'une appli locale connue

    @staticmethod
    def _resolve_real_executable(entry: AppEntry) -> str:
        """
        Beaucoup d'applis (Discord, Slack, GitHub Desktop, WhatsApp
        Desktop...) sont installées via Squirrel : leur clé de registre
        pointe vers `Update.exe`, un lanceur/installeur, PAS l'exécutable
        réel. Le lancer directement sans argument ne fait rien de visible
        (il se ferme aussitôt) -- symptôme typique : "aucune fenêtre ne
        s'ouvre, aucune erreur".

        Le vrai exécutable vit dans un sous-dossier `app-x.y.z\\` à côté
        d'Update.exe. On prend la version la plus récente (tri des noms
        de dossier) et on cherche dedans un .exe qui n'est pas lui-même
        un utilitaire technique (Update.exe, unins*.exe, vk_swiftshader,
        etc.).
        """
        path = Path(entry.exe_path)
        if path.name.lower() != "update.exe":
            return entry.exe_path

        parent = path.parent
        app_folders = sorted(
            (p for p in parent.glob("app-*") if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,  # dernière version en premier (tri lexicographique -- suffisant ici)
        )

        _IGNORE_NAMES = {"update.exe", "vk_swiftshader.exe", "vulkan-1.dll"}
        for folder in app_folders:
            candidates = [
                f for f in folder.glob("*.exe")
                if f.name.lower() not in _IGNORE_NAMES
                and "uninstall" not in f.name.lower()
            ]
            if candidates:
                # Priorité au .exe dont le nom se rapproche le plus du nom
                # affiché (ex. "Discord.exe" pour l'appli "Discord").
                target_hint = entry.name.lower().split()[0]
                candidates.sort(key=lambda f: target_hint not in f.name.lower())
                return str(candidates[0])

        # Rien trouvé -- on retente Update.exe avec l'argument Squirrel
        # standard, en dernier recours (peut fonctionner selon l'appli).
        print(f"[AppLauncher] Aucun exécutable réel trouvé sous {parent}, tentative via Update.exe.")
        return entry.exe_path

    def refresh_index(self) -> int:
        entries: dict[str, AppEntry] = {}  # clé = exe_path normalisé, dédoublonnage

        for entry in self._scan_registry():
            entries.setdefault(self._norm(entry.exe_path), entry)
        for entry in self._scan_start_menu():
            entries[self._norm(entry.exe_path)] = entry  # le Menu Démarrer a de meilleurs noms -- écrase
        for entry in self._scan_folders():
            entries.setdefault(self._norm(entry.exe_path), entry)

        self._index = list(entries.values())
        self._save_cache()
        print(f"[AppLauncher] Index reconstruit : {len(self._index)} applications trouvées.")
        return len(self._index)

    def list_all(self) -> List[AppEntry]:
        return list(self._index)

    # ------------------------------------------------------------------
    # Scan registre
    # ------------------------------------------------------------------
    def _scan_registry(self) -> List[AppEntry]:
        results = []
        for hive, subkey_path in _UNINSTALL_REGISTRY_ROOTS:
            try:
                root = winreg.OpenKey(hive, subkey_path)
            except OSError:
                continue
            for i in range(winreg.QueryInfoKey(root)[0]):
                try:
                    subkey_name = winreg.EnumKey(root, i)
                    with winreg.OpenKey(root, subkey_name) as key:
                        entry = self._registry_entry_from_key(key)
                        if entry:
                            results.append(entry)
                except OSError:
                    continue
            winreg.CloseKey(root)
        return results

    @staticmethod
    def _registry_entry_from_key(key) -> Optional[AppEntry]:
        try:
            name = winreg.QueryValueEx(key, "DisplayName")[0]
        except FileNotFoundError:
            return None
        if not name:
            return None

        exe_path = None

        # DisplayIcon pointe souvent directement vers l'exécutable
        # principal (parfois suivi de ",0" -- index d'icône à retirer).
        try:
            icon = winreg.QueryValueEx(key, "DisplayIcon")[0]
            icon = icon.split(",")[0].strip('"')
            if icon.lower().endswith(".exe") and os.path.exists(icon):
                exe_path = icon
        except FileNotFoundError:
            pass

        # Sinon, on cherche un .exe dans InstallLocation.
        if exe_path is None:
            try:
                install_dir = winreg.QueryValueEx(key, "InstallLocation")[0]
                if install_dir and os.path.isdir(install_dir):
                    for f in os.listdir(install_dir):
                        if f.lower().endswith(".exe"):
                            exe_path = os.path.join(install_dir, f)
                            break
            except FileNotFoundError:
                pass

        if exe_path is None:
            return None
        return AppEntry(name=name, exe_path=exe_path, source="registry")

    # ------------------------------------------------------------------
    # Scan Menu Démarrer (.lnk)
    # ------------------------------------------------------------------
    def _scan_start_menu(self) -> List[AppEntry]:
        if not _HAS_PYWIN32:
            print("[AppLauncher] pywin32 non installé -- Menu Démarrer ignoré (pip install pywin32).")
            return []

        results = []
        shell = win32com.client.Dispatch("WScript.Shell")
        for folder in _START_MENU_FOLDERS:
            if not os.path.isdir(folder):
                continue
            for lnk_path in Path(folder).rglob("*.lnk"):
                try:
                    shortcut = shell.CreateShortCut(str(lnk_path))
                    target = shortcut.Targetpath
                    if target and target.lower().endswith(".exe") and os.path.exists(target):
                        name = lnk_path.stem
                        results.append(AppEntry(name=name, exe_path=target, source="start_menu"))
                except Exception:
                    continue
        return results

    # ------------------------------------------------------------------
    # Scan dossiers standards (filet de sécurité)
    # ------------------------------------------------------------------
    def _scan_folders(self) -> List[AppEntry]:
        results = []
        for base in _STANDARD_FOLDERS:
            if not base or not os.path.isdir(base):
                continue
            base_depth = base.rstrip(os.sep).count(os.sep)
            for root, dirs, files in os.walk(base):
                depth = root.rstrip(os.sep).count(os.sep) - base_depth
                if depth >= _FOLDER_SCAN_MAX_DEPTH:
                    dirs[:] = []  # ne descend pas plus loin
                    continue
                for f in files:
                    if f.lower().endswith(".exe"):
                        exe_path = os.path.join(root, f)
                        name = Path(f).stem.replace("_", " ").replace("-", " ").title()
                        results.append(AppEntry(name=name, exe_path=exe_path, source="folder_scan"))
        return results

    # ------------------------------------------------------------------
    # Cache JSON
    # ------------------------------------------------------------------
    def _save_cache(self):
        data = [{"name": e.name, "exe_path": e.exe_path, "source": e.source} for e in self._index]
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[AppLauncher] Impossible d'écrire le cache ({e}) -- non bloquant.")

    def _load_cache(self):
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._index = [AppEntry(**item) for item in data]
            print(f"[AppLauncher] {len(self._index)} applications chargées depuis le cache.")
        except (OSError, json.JSONDecodeError, TypeError) as e:
            print(f"[AppLauncher] Cache invalide ({e}), réindexation...")
            self.refresh_index()

    @staticmethod
    def _norm(path: str) -> str:
        return os.path.normcase(os.path.normpath(path))
