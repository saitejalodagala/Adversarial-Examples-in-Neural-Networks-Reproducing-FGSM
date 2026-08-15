import inspect
from typing import Dict, Optional, Type
import torch
import torch.nn as nn

from adv_studio.attacks.base import BaseAttack
from adv_studio.attacks.carlini_wagner import CarliniWagnerL2Attack
from adv_studio.attacks.deepfool import DeepFoolAttack
from adv_studio.attacks.fgsm import FGSMAttack, TargetedFGSMAttack
from adv_studio.attacks.ifgsm import IFGSMAttack
from adv_studio.attacks.momentum_fgsm import MomentumFGSMAttack
from adv_studio.attacks.pgd import PGDAttack
from adv_studio.attacks.random_noise import RandomNoiseAttack

ATTACK_REGISTRY: Dict[str, Type[BaseAttack]] = {
    "fgsm": FGSMAttack,
    "targeted_fgsm": TargetedFGSMAttack,
    "ifgsm": IFGSMAttack,
    "bim": IFGSMAttack,
    "pgd": PGDAttack,
    "pgd_linf": PGDAttack,
    "cw": CarliniWagnerL2Attack,
    "carlini_wagner": CarliniWagnerL2Attack,
    "deepfool": DeepFoolAttack,
    "mifgsm": MomentumFGSMAttack,
    "mi_fgsm": MomentumFGSMAttack,
    "random_noise": RandomNoiseAttack,
    "noise": RandomNoiseAttack,
}


def get_attack(
    name: str,
    model: nn.Module,
    epsilon: float = 0.2,
    device: Optional[torch.device] = None,
    **kwargs,
) -> BaseAttack:
    """
    Attack factory function to instantiate any attack algorithm by string name,
    automatically filtering kwargs to match the attack constructor's signature.
    """
    key = name.lower().strip()
    if key not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack '{name}'. Available: {list(ATTACK_REGISTRY.keys())}")

    attack_cls = ATTACK_REGISTRY[key]
    sig = inspect.signature(attack_cls.__init__)
    valid_params = sig.parameters.keys()

    # Base arguments
    call_kwargs = {"model": model, "device": device}
    if "epsilon" in valid_params:
        call_kwargs["epsilon"] = epsilon

    # Filter additional kwargs
    for k, v in kwargs.items():
        if k in valid_params and v is not None:
            call_kwargs[k] = v

    return attack_cls(**call_kwargs)
