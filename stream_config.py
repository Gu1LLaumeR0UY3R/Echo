# -*- coding: utf-8 -*-
"""
Configuration du pipeline RVC en streaming temps réel.

Les paramètres et la logique de calcul des tailles de fenêtres reprennent
exactement ceux du mode temps réel officiel de RVC
(RVCRealtimeVST/worker/rvc_worker.py, classe RVCStreamEngine.__init__),
récupérés depuis le repo GitHub RVC-Project/Retrieval-based-Voice-Conversion-WebUI
le 12/08/2026. Seule l'implémentation de l'inférence change (on branche
notre propre hubert_engine / rmvpe_engine au lieu de infer.rtrvc.RVC).

Vocabulaire :
- block      : le morceau audio "nouveau" qu'on reçoit à chaque itération
               (ex: 200ms). C'est ce qu'on veut convertir et jouer.
- crossfade  : recouvrement entre la fin du bloc précédent et le début du
               bloc courant, utilisé pour fondre les deux sans "clic".
- extra      : contexte passé supplémentaire donné à HuBERT/RMVPE pour
               qu'ils aient assez d'historique pour bien extraire le
               contenu phonétique et la pitch (ces modèles ne fonctionnent
               pas bien sur des fenêtres trop courtes isolées).
- zc         : "zero crossing" unit = sample_rate // 100, soit 10ms.
               Toutes les fenêtres sont arrondies à ce multiple pour rester
               alignées avec le sous-échantillonnage 16kHz utilisé par
               HuBERT/RMVPE.
"""
from dataclasses import dataclass, field


@dataclass
class StreamConfig:
    sample_rate: int = 40000        # taux du modèle RVC (40000 pour la plupart des .pth v2, vérifier le vôtre)
    block_ms: float = 200.0         # durée d'un bloc traité à chaque itération
    crossfade_ms: float = 40.0      # recouvrement demandé (sera plafonné, voir sola_buffer_frame)
    extra_ms: float = 500.0         # contexte passé fourni à HuBERT/RMVPE
    f0_method: str = "rmvpe"        # "rmvpe" (celui que vous utilisez déjà) ou "fcpe"/"pm" si besoin plus rapide

    def __post_init__(self):
        self.zc = max(1, self.sample_rate // 100)

        self.block_frame = self._round_to_zc(self.block_ms)
        self.block_frame_16k = 160 * self.block_frame // self.zc

        self.crossfade_frame = self._round_to_zc(self.crossfade_ms)
        # RVC officiel plafonne le buffer SOLA à 40ms (4 * zc), indépendamment
        # de block_ms. Au-delà, ça n'améliore pas la qualité et ça coûte du
        # temps de calcul supplémentaire à chaque bloc.
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.effective_crossfade_ms = 1000.0 * self.sola_buffer_frame / self.sample_rate
        self.sola_search_frame = self.zc

        self.extra_frame = self._round_to_zc(self.extra_ms)
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc

    def _round_to_zc(self, ms: float) -> int:
        return int(round(ms / 1000 * self.sample_rate / self.zc) * self.zc)

    def summary(self) -> str:
        return (
            f"block={self.block_ms:.0f}ms ({self.block_frame} frames) | "
            f"crossfade effectif={self.effective_crossfade_ms:.0f}ms | "
            f"contexte={self.extra_ms:.0f}ms | "
            f"latence théorique ~= block + crossfade = "
            f"{self.block_ms + self.effective_crossfade_ms:.0f}ms (hors temps de calcul GPU)"
        )


if __name__ == "__main__":
    cfg = StreamConfig()
    print(cfg.summary())
