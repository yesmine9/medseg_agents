"""État partagé du graphe. Les agents ne voient JAMAIS la vérité terrain."""
from typing import TypedDict
import numpy as np


class AgentState(TypedDict, total=False):
    stem: str
    image: np.ndarray        # RGB uint8 HxWx3
    attempt: int             # 0 = premier passage, 1..max_retries = re-segmentations
    strategy: str            # base | tta | clahe
    mask: np.ndarray         # masque de la tentative courante (bool HxW)
    features: dict           # mesures du critic pour la tentative courante
    risk: float              # score de risque (0 = sûr, 1 = très douteux)
    decision: str            # accept | retry | escalate
    best: dict               # meilleure tentative (risque minimal) : mask, risk, attempt, features
    attempts: list           # toutes les tentatives : attempt, strategy, risk, features, mask
    history: list            # trace lisible de toutes les décisions
    final_mask: np.ndarray
    status: str              # accepted | accepted_after_retry | needs_human
