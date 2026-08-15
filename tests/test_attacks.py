import pytest
import torch
import torch.nn as nn

from adv_studio.models import SimpleCNN
from adv_studio.attacks import (
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


@pytest.fixture
def mock_model():
    model = SimpleCNN()
    model.eval()
    return model


@pytest.fixture
def sample_batch():
    images = torch.rand(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    return images, labels


def test_fgsm_bounds(mock_model, sample_batch):
    images, labels = sample_batch
    epsilon = 0.25
    attack = FGSMAttack(mock_model, epsilon=epsilon)
    adv_images = attack.generate(images, labels)

    assert adv_images.shape == images.shape
    assert torch.all(adv_images >= 0.0)
    assert torch.all(adv_images <= 1.0)

    # Check L_inf perturbation bound
    diff = torch.abs(adv_images - images)
    assert torch.max(diff).item() <= epsilon + 1e-5


def test_targeted_fgsm(mock_model, sample_batch):
    images, _ = sample_batch
    targets = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    attack = TargetedFGSMAttack(mock_model, epsilon=0.2)
    adv_images = attack.generate(images, labels=targets, target_labels=targets)

    assert adv_images.shape == images.shape
    assert torch.all(adv_images >= 0.0)
    assert torch.all(adv_images <= 1.0)


def test_ifgsm_bounds(mock_model, sample_batch):
    images, labels = sample_batch
    epsilon = 0.2
    attack = IFGSMAttack(mock_model, epsilon=epsilon, steps=5)
    adv_images = attack.generate(images, labels)

    assert adv_images.shape == images.shape
    assert torch.all(adv_images >= 0.0)
    assert torch.all(adv_images <= 1.0)
    diff = torch.abs(adv_images - images)
    assert torch.max(diff).item() <= epsilon + 1e-5


def test_pgd_linf_bounds(mock_model, sample_batch):
    images, labels = sample_batch
    epsilon = 0.2
    attack = PGDAttack(mock_model, epsilon=epsilon, steps=5, norm="Linf")
    adv_images = attack.generate(images, labels)

    assert adv_images.shape == images.shape
    assert torch.all(adv_images >= 0.0)
    assert torch.all(adv_images <= 1.0)
    diff = torch.abs(adv_images - images)
    assert torch.max(diff).item() <= epsilon + 1e-5


def test_momentum_fgsm(mock_model, sample_batch):
    images, labels = sample_batch
    epsilon = 0.2
    attack = MomentumFGSMAttack(mock_model, epsilon=epsilon, steps=5)
    adv_images = attack.generate(images, labels)

    assert adv_images.shape == images.shape
    assert torch.all(adv_images >= 0.0)
    assert torch.all(adv_images <= 1.0)


def test_carlini_wagner_l2(mock_model, sample_batch):
    images, labels = sample_batch
    attack = CarliniWagnerL2Attack(mock_model, steps=5, lr=0.05)
    adv_images = attack.generate(images, labels)

    assert adv_images.shape == images.shape
    assert torch.all(adv_images >= 0.0)
    assert torch.all(adv_images <= 1.0)


def test_deepfool(mock_model):
    # Single sample test for deepfool
    images = torch.rand(1, 1, 28, 28)
    labels = torch.tensor([0], dtype=torch.long)
    attack = DeepFoolAttack(mock_model, max_iter=5)
    adv_images = attack.generate(images, labels)

    assert adv_images.shape == images.shape
    assert torch.all(adv_images >= 0.0)
    assert torch.all(adv_images <= 1.0)


def test_random_noise(mock_model, sample_batch):
    images, labels = sample_batch
    attack = RandomNoiseAttack(mock_model, epsilon=0.2)
    adv_images = attack.generate(images, labels)

    assert adv_images.shape == images.shape
    assert torch.all(adv_images >= 0.0)
    assert torch.all(adv_images <= 1.0)


def test_attack_factory(mock_model):
    attack = get_attack("pgd", mock_model, epsilon=0.15)
    assert isinstance(attack, PGDAttack)
    assert attack.epsilon == 0.15
