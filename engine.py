# -*- coding: utf-8 -*-
"""
engine.py
=========
Point d'entrée "haut niveau" du moteur de conversion vocale.

C'est ce fichier qu'Echo (ou notre script de test) doit utiliser. Il cache
toute la complexité (chargement HuBERT, RMVPE, réseau de synthèse) derrière
une interface simple :

    voice_engine = VoiceEngine(config)
    voice_engine.load_voice_model("miku_mellow_rvc.pth")
    audio_de_sortie = voice_engine.convert(audio_neutre, pitch_shift=10)

Le modèle HuBERT et RMVPE sont chargés UNE SEULE FOIS au démarrage (pas à
chaque phrase), et restent en mémoire tant que le programme tourne. C'est
exactement ce qui corrige le problème de lenteur (2 minutes par appel) de
l'ancienne architecture en sous-process.
"""

from pathlib import Path
import torch

from arch.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from hubert_engine import load_hubert_model
from rmvpe_engine import RMVPE
from pipeline import Pipeline

# Associe (version, présence de pitch) à la bonne classe de réseau.
# C'est un détail d'architecture : un modèle "v2 avec pitch" (Miku) n'a pas
# exactement la même structure de couches qu'un modèle "v1 sans pitch" (BT).
_SYNTH_CLASSES = {
    ("v1", True): SynthesizerTrnMs256NSFsid,
    ("v1", False): SynthesizerTrnMs256NSFsid_nono,
    ("v2", True): SynthesizerTrnMs768NSFsid,
    ("v2", False): SynthesizerTrnMs768NSFsid_nono,
}


class VoiceEngine:
    def __init__(self, config, hubert_pt_path: str, rmvpe_path: str):
        """
        config          : instance de config.Config (device, précision, etc.)
        hubert_pt_path  : chemin vers le fichier hubert_base.pt (modèle brut
                          fairseq, converti automatiquement au chargement)
        rmvpe_path      : chemin vers rmvpe.pt
        """
        self.config = config
        self.device = config.device

        print("[engine] Chargement de HuBERT (contenu vocal)...")
        self.hubert_model = load_hubert_model(hubert_pt_path, self.device, config.is_half)

        print("[engine] Chargement de RMVPE (détection du pitch)...")
        self.rmvpe_model = RMVPE(rmvpe_path, is_half=config.is_half, device=self.device)

        # État du modèle vocal actuellement chargé (Miku, BT-7274, ou aucun).
        self.net_g = None
        self.pipeline = None
        self.target_sr = None
        self.has_f0 = None
        self.version = None
        self.current_model_name = None

    def load_voice_model(self, pth_path: str):
        """
        Charge un modèle vocal (.pth) en mémoire GPU/CPU.
        À appeler une fois par voix, puis on peut faire autant de conversions
        que voulu sans recharger.
        """
        pth_path = str(pth_path)
        print(f"[engine] Chargement du modèle vocal : {pth_path}")
        checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)

        self.target_sr = checkpoint["config"][-1]
        # Nombre de locuteurs déduit des poids réels (robustesse).
        checkpoint["config"][-3] = checkpoint["weight"]["emb_g.weight"].shape[0]

        self.has_f0 = bool(checkpoint.get("f0", 1))
        self.version = checkpoint.get("version", "v1")

        synth_class = _SYNTH_CLASSES[(self.version, self.has_f0)]
        self.net_g = synth_class(*checkpoint["config"], is_half=self.config.is_half)

        # enc_q n'est utilisé que pendant l'entraînement, inutile en inférence.
        del self.net_g.enc_q
        self.net_g.load_state_dict(checkpoint["weight"], strict=False)
        self.net_g = self.net_g.eval().to(self.device)
        self.net_g = self.net_g.half() if self.config.is_half else self.net_g.float()

        self.pipeline = Pipeline(self.target_sr, self.config)
        self.current_model_name = Path(pth_path).stem

        print(
            f"[engine] Modèle chargé : {self.current_model_name} "
            f"(version={self.version}, pitch={'oui' if self.has_f0 else 'non'}, "
            f"sample_rate={self.target_sr})"
        )

    def convert(
        self,
        audio_16k,
        pitch_shift: int = 0,
        index_path: str = "",
        index_rate: float = 0.75,
        protect: float = 0.33,
    ):
        """
        Convertit un audio (déjà chargé à 16000 Hz, voir audio_io.load_audio)
        vers la voix actuellement chargée.

        pitch_shift : décalage en demi-tons (ignoré si le modèle n'a pas de
                      pitch, comme BT-7274).
        index_path  : chemin vers le fichier .index (optionnel mais
                      recommandé, améliore la fidélité au timbre).
        index_rate  : dosage du retrieval (0 = désactivé, 1 = max).
        protect     : protège les consonnes/silences de la distorsion du
                      retrieval (0.33 = valeur par défaut RVC).
        """
        if self.net_g is None:
            raise RuntimeError(
                "Aucun modèle vocal chargé. Appelle load_voice_model() d'abord."
            )

        return self.pipeline.run(
            hubert_model=self.hubert_model,
            rmvpe_model=self.rmvpe_model,
            net_g=self.net_g,
            audio=audio_16k,
            pitch_shift_semitones=pitch_shift,
            file_index_path=index_path,
            index_rate=index_rate,
            has_f0=self.has_f0,
            target_sample_rate=self.target_sr,
            protect=protect,
            version=self.version,
        )
