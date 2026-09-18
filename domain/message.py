# -*- coding: utf-8 -*-
"""
domain/message.py
==================
Le cœur métier (couche "domain" de l'architecture hexagonale) : ne dépend
d'AUCUN adapter, d'AUCUNE librairie externe (Qwen, RVC, GUI...). C'est la
définition la plus neutre possible de ce qu'est un échange dans une
conversation avec Echo.
"""
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    role: str          # "user" (vous) ou "assistant" (Echo/Miku)
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
