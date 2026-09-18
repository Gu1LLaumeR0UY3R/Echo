# -*- coding: utf-8 -*-
"""
hubert_engine.py
=================
BRIQUE 1 du pipeline : extraction du "contenu vocal".

Rôle : prendre l'audio brut (ce qui a été dit, avec la voix neutre d'Edge-TTS)
et en extraire une représentation numérique qui capture UNIQUEMENT le contenu
phonétique (les sons, les mots prononcés) — PAS le timbre de la voix d'origine.

C'est cette représentation "neutre" qui sera ensuite injectée dans le modèle
Miku ou BT-7274 pour être reconvertie avec leur timbre à eux.

---------------------------------------------------------------------------
POINT TECHNIQUE IMPORTANT : pourquoi ce fichier convertit le modèle lui-même
---------------------------------------------------------------------------
Le fichier `hubert_base.pt` que tu télécharges (depuis HuggingFace, projet
lj1995/VoiceConversionWebUI) est au format "fairseq" — l'ancienne librairie
de Facebook Research utilisée pour entraîner HuBERT à l'origine. Pour LIRE ce
format normalement, il faut installer `fairseq`, une librairie connue pour
être pénible à installer (extensions C++, dépendances cassées, en particulier
sur Windows) — exactement le genre de dépendance qu'on cherche à éviter.

Solution : ce fichier lit les poids bruts du fichier `.pt` (juste des
tenseurs, aucune exécution de code fairseq), puis les RENOMME pour qu'ils
correspondent au format attendu par `transformers` (la librairie HuggingFace,
beaucoup plus simple à installer). C'est une conversion de nommage uniquement
— les poids eux-mêmes, donc "l'intelligence" du modèle, ne changent pas d'un
seul bit. Vérifié : après conversion, 0 clé manquante, 0 clé en trop.

Résultat concret : on ne dépend QUE de `torch` + `transformers`, jamais de
`fairseq`. Un seul fichier `hubert_base.pt` suffit, pas de dossier HuggingFace
à préparer à la main.
"""

import logging
import pickle
import types

import torch
from torch import nn
from transformers import HubertConfig, HubertModel

logger = logging.getLogger(__name__)

# Hyperparamètres du modèle "HuBERT base" utilisé par RVC (12 couches,
# 768 dimensions). Ce sont des constantes fixes, communes à tous les modèles
# RVC v1 et v2 -- ce n'est pas un réglage à modifier.
_HUBERT_CONFIG = HubertConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    conv_dim=(512, 512, 512, 512, 512, 512, 512),
    conv_stride=(5, 2, 2, 2, 2, 2, 2),
    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
    conv_bias=False,
    feat_extract_norm="group",
    do_stable_layer_norm=False,
    classifier_proj_size=256,
)


class HubertModelWithFinalProj(HubertModel):
    """
    HuBERT standard + une couche de projection finale.
    Cette couche supplémentaire est ce que RVC a appris pendant l'entraînement
    des modèles de contenu ; elle fait partie du modèle "hubert_base" fourni
    par le projet RVC (pas un modèle HuBERT générique tout nu).
    """

    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


class _TolerantUnpickler(pickle.Unpickler):
    """
    Unpickler "tolérant" : quand il tombe sur une classe fairseq qu'on n'a
    pas installée (ex: fairseq.data.dictionary.Dictionary), il la remplace
    par une classe vide au lieu de planter. On n'a besoin que des TENSEURS
    de poids (dans la clé "model" du checkpoint), pas du reste (config
    d'entraînement, dictionnaire de vocabulaire, etc.) donc ça ne pose aucun
    problème de les ignorer.
    """

    class _Placeholder:
        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)

        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {})

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return self._Placeholder


def _load_raw_fairseq_checkpoint(pt_path: str) -> dict:
    """Charge le fichier .pt fairseq et retourne uniquement son state_dict brut."""
    fake_pickle_module = types.ModuleType("fake_pickle_module")
    fake_pickle_module.Unpickler = _TolerantUnpickler
    fake_pickle_module.load = pickle.load
    fake_pickle_module.dump = pickle.dump
    fake_pickle_module.HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL

    with open(pt_path, "rb") as f:
        checkpoint = torch.load(
            f, map_location="cpu", weights_only=False, pickle_module=fake_pickle_module
        )
    return checkpoint["model"]


def _convert_fairseq_to_transformers(fairseq_sd: dict) -> dict:
    """Renomme les clés du format fairseq vers le format transformers (HF)."""
    hf_sd = {"masked_spec_embed": fairseq_sd["mask_emb"]}

    for i in range(7):
        hf_sd[f"feature_extractor.conv_layers.{i}.conv.weight"] = fairseq_sd[
            f"feature_extractor.conv_layers.{i}.0.weight"
        ]
    hf_sd["feature_extractor.conv_layers.0.layer_norm.weight"] = fairseq_sd[
        "feature_extractor.conv_layers.0.2.weight"
    ]
    hf_sd["feature_extractor.conv_layers.0.layer_norm.bias"] = fairseq_sd[
        "feature_extractor.conv_layers.0.2.bias"
    ]

    hf_sd["feature_projection.projection.weight"] = fairseq_sd["post_extract_proj.weight"]
    hf_sd["feature_projection.projection.bias"] = fairseq_sd["post_extract_proj.bias"]
    hf_sd["feature_projection.layer_norm.weight"] = fairseq_sd["layer_norm.weight"]
    hf_sd["feature_projection.layer_norm.bias"] = fairseq_sd["layer_norm.bias"]

    hf_sd["encoder.pos_conv_embed.conv.bias"] = fairseq_sd["encoder.pos_conv.0.bias"]
    hf_sd["encoder.pos_conv_embed.conv.parametrizations.weight.original0"] = fairseq_sd[
        "encoder.pos_conv.0.weight_g"
    ]
    hf_sd["encoder.pos_conv_embed.conv.parametrizations.weight.original1"] = fairseq_sd[
        "encoder.pos_conv.0.weight_v"
    ]
    hf_sd["encoder.layer_norm.weight"] = fairseq_sd["encoder.layer_norm.weight"]
    hf_sd["encoder.layer_norm.bias"] = fairseq_sd["encoder.layer_norm.bias"]

    for i in range(12):
        src = f"encoder.layers.{i}."
        dst = f"encoder.layers.{i}."
        for proj in ["k_proj", "v_proj", "q_proj", "out_proj"]:
            hf_sd[f"{dst}attention.{proj}.weight"] = fairseq_sd[f"{src}self_attn.{proj}.weight"]
            hf_sd[f"{dst}attention.{proj}.bias"] = fairseq_sd[f"{src}self_attn.{proj}.bias"]
        hf_sd[f"{dst}layer_norm.weight"] = fairseq_sd[f"{src}self_attn_layer_norm.weight"]
        hf_sd[f"{dst}layer_norm.bias"] = fairseq_sd[f"{src}self_attn_layer_norm.bias"]
        hf_sd[f"{dst}feed_forward.intermediate_dense.weight"] = fairseq_sd[f"{src}fc1.weight"]
        hf_sd[f"{dst}feed_forward.intermediate_dense.bias"] = fairseq_sd[f"{src}fc1.bias"]
        hf_sd[f"{dst}feed_forward.output_dense.weight"] = fairseq_sd[f"{src}fc2.weight"]
        hf_sd[f"{dst}feed_forward.output_dense.bias"] = fairseq_sd[f"{src}fc2.bias"]
        hf_sd[f"{dst}final_layer_norm.weight"] = fairseq_sd[f"{src}final_layer_norm.weight"]
        hf_sd[f"{dst}final_layer_norm.bias"] = fairseq_sd[f"{src}final_layer_norm.bias"]

    hf_sd["final_proj.weight"] = fairseq_sd["final_proj.weight"]
    hf_sd["final_proj.bias"] = fairseq_sd["final_proj.bias"]
    return hf_sd


def load_hubert_model(hubert_pt_path: str, device: str, is_half: bool = False):
    """
    Charge le modèle HuBERT directement depuis le fichier `hubert_base.pt`
    téléchargé (format fairseq d'origine), sans dépendance à `fairseq`.

    `hubert_pt_path` : chemin vers le fichier hubert_base.pt (ex: dans le
    dossier models/ du projet, voir README.md).
    """
    logger.info("Chargement de HuBERT depuis %s", hubert_pt_path)

    fairseq_sd = _load_raw_fairseq_checkpoint(hubert_pt_path)
    hf_sd = _convert_fairseq_to_transformers(fairseq_sd)

    model = HubertModelWithFinalProj(_HUBERT_CONFIG)
    missing = model.load_state_dict(hf_sd, strict=True)  # doit être parfait
    del missing  # strict=True lève une exception sinon, rien à vérifier de plus

    dtype = torch.float16 if is_half else torch.float32
    model = model.to(device=device, dtype=dtype)
    return model.eval()


def extract_hubert_features(model, source: torch.Tensor, version: str, padding_mask=None):
    """
    Fait passer l'audio dans HuBERT et retourne la représentation de contenu.

    `version` détermine quelle "couche" du réseau on utilise :
      - "v1" (256 dimensions) : couche 9 + projection finale
      - "v2" (768 dimensions) : dernière couche (12) directement
    C'est déterminé par le modèle vocal (Miku = v2, BT-7274 = v1), pas un
    choix arbitraire : chaque modèle a été entraîné avec l'une ou l'autre.
    """
    if version not in {"v1", "v2"}:
        raise ValueError(f"Version RVC non supportée : {version!r}")

    attention_mask = None
    if padding_mask is not None and bool(torch.any(padding_mask).item()):
        attention_mask = (~padding_mask.bool()).long()

    with torch.no_grad():
        if version == "v1":
            outputs = model(
                input_values=source,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            return model.final_proj(outputs.hidden_states[9])
        else:
            outputs = model(
                input_values=source,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            return outputs.last_hidden_state
