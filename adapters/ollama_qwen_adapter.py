# -*- coding: utf-8 -*-
"""
adapters/ollama_qwen_adapter.py
=================================
Implémentation concrète de TextGenerationPort avec Qwen tournant en local
via Ollama. Utilise l'endpoint /api/chat (pas /api/generate comme dans les
scripts de test précédents) pour bénéficier de la vraie mémoire de
conversation multi-tours gérée nativement par Ollama.
"""
from typing import List

import requests

from domain.message import Message
from ports.text_generation_port import TextGenerationPort


class OllamaQwenAdapter(TextGenerationPort):
    def __init__(
        self,
        model: str,
        system_prompt: str,
        max_tokens: int = 60,
        url: str = "http://localhost:11434/api/chat",
        timeout_s: int = 120,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.url = url
        self.timeout_s = timeout_s

    def generate(self, history: List[Message]) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        for m in history:
            messages.append({"role": m.role, "content": m.content})

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": self.max_tokens},
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
