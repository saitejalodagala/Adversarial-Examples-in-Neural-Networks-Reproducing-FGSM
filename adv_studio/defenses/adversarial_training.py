from typing import Callable, Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from adv_studio.attacks.fgsm import FGSMAttack
from adv_studio.attacks.pgd import PGDAttack


class AdversarialTrainer:
    """
    Unified Adversarial Training framework implementing:
    - Goodfellow FGSM adversarial training (mixed loss: alpha * L_clean + (1 - alpha) * L_adv)
    - Madry PGD adversarial training: min_theta E [ max_delta in S L(theta, x + delta, y) ]
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        attack_type: str = "fgsm",
        epsilon: float = 0.2,
        alpha_mix: float = 0.5,
        pgd_steps: int = 10,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.attack_type = attack_type.lower()
        self.epsilon = float(epsilon)
        self.alpha_mix = float(alpha_mix)
        self.pgd_steps = int(pgd_steps)
        self.device = device or (next(model.parameters()).device if list(model.parameters()) else torch.device("cpu"))

        if self.attack_type == "pgd":
            self.attack = PGDAttack(model=self.model, epsilon=self.epsilon, steps=self.pgd_steps, device=self.device)
        else:
            self.attack = FGSMAttack(model=self.model, epsilon=self.epsilon, device=self.device)

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float, float]:
        """
        Run one epoch of adversarial training.

        Returns:
            (avg_loss, clean_accuracy, adversarial_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        clean_correct = 0
        adv_correct = 0
        total_samples = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # Generate adversarial examples for the batch
            adv_images = self.attack.generate(images, labels)

            self.optimizer.zero_grad()

            outputs_clean = self.model(images)
            outputs_adv = self.model(adv_images)

            loss_clean = self.criterion(outputs_clean, labels)
            loss_adv = self.criterion(outputs_adv, labels)

            # Mixed adversarial loss
            loss = self.alpha_mix * loss_clean + (1.0 - self.alpha_mix) * loss_adv

            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            clean_correct += (outputs_clean.argmax(1) == labels).sum().item()
            adv_correct += (outputs_adv.argmax(1) == labels).sum().item()
            total_samples += batch_size

        return total_loss / total_samples, clean_correct / total_samples, adv_correct / total_samples


class FastAdversarialTrainer:
    """
    Fast Adversarial Training with Random Initialization (Wong et al., ICLR 2020).
    Allows N-step robust training at 1-step cost without catastrophic overfitting.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        epsilon: float = 0.2,
        alpha: float = 0.25,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.device = device or (next(model.parameters()).device if list(model.parameters()) else torch.device("cpu"))

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # 1. Random uniform initialization
            delta = torch.zeros_like(images).uniform_(-self.epsilon, self.epsilon).to(self.device)
            delta.data = torch.clamp(images + delta.data, 0.0, 1.0) - images
            delta.requires_grad = True

            # 2. Fast 1-step gradient calculation
            outputs = self.model(images + delta)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # 3. FGSM step with step size alpha and projection
            grad = delta.grad.detach()
            delta.data = torch.clamp(delta.data + self.alpha * torch.sign(grad), -self.epsilon, self.epsilon)
            delta.data = torch.clamp(images + delta.data, 0.0, 1.0) - images

            self.optimizer.zero_grad()
            final_outputs = self.model(images + delta)
            final_loss = self.criterion(final_outputs, labels)
            final_loss.backward()
            self.optimizer.step()

            total_loss += final_loss.item() * images.size(0)
            correct += (final_outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total
