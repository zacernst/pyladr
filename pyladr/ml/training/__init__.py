"""Training infrastructure for clause embedding models."""

from pyladr.ml.training.contrastive import (
    ContrastiveConfig,
    ContrastiveLoss,
    ContrastiveTrainer,
    EmbeddingEvaluator,
    InferencePair,
    PairLabel,
    ProofPatternExtractor,
    TrainingDataset,
)
from pyladr.ml.training.online_losses import (
    CombinedOnlineLoss,
    LossStatistics,
    OnlineInfoNCELoss,
    OnlineLossConfig,
    OnlineTripletLoss,
)

__all__ = [
    "CombinedOnlineLoss",
    "ContrastiveConfig",
    "ContrastiveLoss",
    "ContrastiveTrainer",
    "EmbeddingEvaluator",
    "InferencePair",
    "LossStatistics",
    "OnlineInfoNCELoss",
    "OnlineLossConfig",
    "OnlineTripletLoss",
    "PairLabel",
    "ProofPatternExtractor",
    "TrainingDataset",
]
