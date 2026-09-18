# -*- coding: utf-8 -*-
"""
streaming_config_rvc.py
========================
Petite config minimale pour instancier VoiceEngine dans les tests de
streaming. Je n'ai pas le contenu exact de votre `config.py` original
(celui utilisé par `rvc_lite_test_code/test.py`), donc si vous en avez
déjà un qui fonctionne, UTILISEZ-LE À LA PLACE — il suffit qu'il expose
les mêmes attributs (`device`, `is_half`, `x_pad`, `x_query`, `x_center`,
`x_max`).

Les valeurs x_pad/x_query/x_center/x_max ci-dessous ne servent QUE si
`Pipeline.run()` est appelé (mode fichier complet, non-streaming) — elles
sont reprises telles quelles pour compatibilité mais ne sont PAS utilisées
par le chemin de streaming (make_lite_engine_infer_fn n'appelle jamais
pipeline.run(), seulement pipeline._extract_pitch()).
"""
import torch


class StreamingRVCConfig:
    def __init__(self, is_half: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # is_half=True (précision float16) recommandé sur RTX 4070 pour la
        # vitesse. Si vous obtenez des NaN/erreurs de dtype, repassez à False.
        self.is_half = is_half and self.device == "cuda"

        # Valeurs par défaut RVC officiel pour la tranche "GPU >= 6-8GB VRAM"
        # (config_data du RVC-Project). Non utilisées en mode streaming.
        self.x_pad = 1
        self.x_query = 6
        self.x_center = 38
        self.x_max = 41
