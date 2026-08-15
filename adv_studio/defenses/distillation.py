from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class DefensiveDistillation:
    """
    Defensive Distillation from Papernot et al. (2016)
    "Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks".

    Mechanism:
    1. Train Teacher network with standard cross entropy at temperature T.
    2. Compute soft probability predictions with temperature T: q_i = exp(z_i / T) / sum(exp(z_j / T)).
    3. Train Student network using soft targets to smooth the gradients and mask sharp decision transitions.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 20.0,
        device: Optional[torch.device] = None,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = float(temperature)
        self.device = device or (next(student_model.parameters()).device if list(student_model.parameters()) else torch.device("cpu"))

    def train_student_epoch(self, loader: DataLoader, optimizer: optim.Optimizer) -> Tuple[float, float]:
        self.teacher.eval()
        self.student.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # Get teacher soft targets
            with torch.no_grad():
                teacher_logits = self.teacher(images)
                soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=1)

            # Student prediction with temperature
            student_logits = self.student(images)
            log_student_probs = nn.functional.log_softmax(student_logits / self.temperature, dim=1)

            # KL-Divergence / Soft cross entropy loss
            loss = nn.functional.kl_div(log_student_probs, soft_targets, reduction="batchmean") * (self.temperature ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (student_logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total
