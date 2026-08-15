"""
Adversarial Studio: Comprehensive Adversarial Machine Learning Library & Interactive Research Platform.
"""

__version__ = "1.0.0"
__author__ = "Sai Teja Lodagala"

from adv_studio.models import SimpleCNN, LeNet5, MiniResNet18, MLP, get_model
from adv_studio.attacks import (
    BaseAttack,
    FGSMAttack,
    TargetedFGSMAttack,
    IFGSMAttack,
    PGDAttack,
    CarliniWagnerL2Attack,
    DeepFoolAttack,
    MomentumFGSMAttack,
    RandomNoiseAttack,
    get_attack,
)
from adv_studio.defenses import (
    AdversarialTrainer,
    FastAdversarialTrainer,
    DefensiveDistillation,
    RandomizedSmoothing,
    BitDepthReduction,
    SpatialSmoothing,
    TotalVariationDenoising,
)
from adv_studio.evaluation import (
    compute_robust_accuracy,
    compute_distortion_metrics,
    compute_transferability_matrix,
    compute_loss_landscape_1d,
)
