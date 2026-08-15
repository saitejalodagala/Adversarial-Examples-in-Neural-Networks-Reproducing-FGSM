#!/usr/bin/env python
"""
Multi-Architecture Model Trainer for Adversarial Machine Learning.
Trains Clean and Adversarially-Trained variants of SimpleCNN, LeNet-5, ResNet-18, and MLP.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adv_studio.models import get_model
from adv_studio.data import get_mnist_loaders
from adv_studio.defenses import AdversarialTrainer

def train_model(model_name, epochs=3, adv_train=False, epsilon=0.2, data_dir="./DATA", output_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Training {model_name} (Adv-Train: {adv_train}, Epsilon: {epsilon}) on {device}")

    train_loader, test_loader = get_mnist_loaders(data_dir=data_dir, batch_size=64)
    model = get_model(name=model_name, device=device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    if adv_train:
        trainer = AdversarialTrainer(model, optimizer, criterion, attack_type="fgsm", epsilon=epsilon, device=device)
        for ep in range(1, epochs + 1):
            loss, c_acc, a_acc = trainer.train_epoch(train_loader)
            print(f"Epoch [{ep}/{epochs}] Loss: {loss:.4f} | Clean Acc: {c_acc*100:.2f}% | Adv Acc: {a_acc*100:.2f}%")
    else:
        for ep in range(1, epochs + 1):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                l = criterion(out, lbls)
                l.backward()
                optimizer.step()
                total_loss += l.item() * imgs.size(0)
                correct += (out.argmax(1) == lbls).sum().item()
                total += lbls.size(0)
            print(f"Epoch [{ep}/{epochs}] Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.2f}%")

    os.makedirs(output_dir, exist_ok=True)
    suffix = "_adv_trained.pth" if adv_train else ".pth"
    save_path = os.path.join(output_dir, f"{model_name}{suffix}")
    torch.save(model.state_dict(), save_path)
    print(f"[+] Saved checkpoint to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train multi-architecture models")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--data-dir", type=str, default="./DATA", help="Path to data")
    args = parser.parse_args()

    models = ["simple_cnn", "lenet", "resnet", "mlp"]
    for m in models:
        # Clean
        train_model(m, epochs=args.epochs, adv_train=False, data_dir=args.data_dir)
        # Adv-Trained
        train_model(m, epochs=args.epochs, adv_train=True, epsilon=0.2, data_dir=args.data_dir)

if __name__ == "__main__":
    main()
