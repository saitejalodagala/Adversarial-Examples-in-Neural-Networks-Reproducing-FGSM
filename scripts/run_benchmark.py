#!/usr/bin/env python
"""
Automated Robustness & Adversarial Distortion Benchmark Runner.
Evaluates Clean Model vs Adversarially Trained Model across all attack suites.
"""

import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adv_studio.models import get_model
from adv_studio.data import get_mnist_loaders
from adv_studio.attacks import (
    FGSMAttack,
    IFGSMAttack,
    PGDAttack,
    CarliniWagnerL2Attack,
    DeepFoolAttack,
    MomentumFGSMAttack,
    RandomNoiseAttack,
)
from adv_studio.evaluation import compute_robust_accuracy

def main():
    parser = argparse.ArgumentParser(description="Run Adversarial Machine Learning Benchmark")
    parser.add_argument("--data-dir", type=str, default="./DATA", help="Path to dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-batches", type=int, default=2, help="Number of test batches")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Benchmark Hardware Device: {device}")

    _, test_loader = get_mnist_loaders(data_dir=args.data_dir, batch_size=args.batch_size)

    clean_path = "checkpoints/mnist_cnn.pth"
    adv_path = "checkpoints/mnist_cnn_adv_trained.pth"

    clean_model = get_model("simple_cnn", pretrained_path=clean_path if os.path.exists(clean_path) else None, device=device)
    adv_model = get_model("simple_cnn", pretrained_path=adv_path if os.path.exists(adv_path) else None, device=device)

    clean_model.eval()
    adv_model.eval()

    attack_configs = [
        ("Clean (No Attack)", lambda m: None),
        ("Random Noise (eps=0.2)", lambda m: RandomNoiseAttack(m, epsilon=0.2, device=device)),
        ("FGSM (eps=0.1)", lambda m: FGSMAttack(m, epsilon=0.1, device=device)),
        ("FGSM (eps=0.2)", lambda m: FGSMAttack(m, epsilon=0.2, device=device)),
        ("FGSM (eps=0.3)", lambda m: FGSMAttack(m, epsilon=0.3, device=device)),
        ("I-FGSM (eps=0.2, T=10)", lambda m: IFGSMAttack(m, epsilon=0.2, steps=10, device=device)),
        ("PGD-Linf (eps=0.2, T=10)", lambda m: PGDAttack(m, epsilon=0.2, steps=10, device=device)),
        ("MI-FGSM (eps=0.2, T=10)", lambda m: MomentumFGSMAttack(m, epsilon=0.2, steps=10, device=device)),
        ("DeepFool (max_iter=10)", lambda m: DeepFoolAttack(m, max_iter=10, device=device)),
        ("Carlini-Wagner L2 (steps=15)", lambda m: CarliniWagnerL2Attack(m, steps=15, device=device)),
    ]

    results = []
    print("\n" + "=" * 80)
    print(f"{'Attack Configuration':<32} | {'Clean Model Acc':<16} | {'Adv Model Acc':<16} | {'Delta':<8}")
    print("=" * 80)

    for name, atk_fn in attack_configs:
        atk_clean = atk_fn(clean_model)
        atk_adv = atk_fn(adv_model)

        res_clean = compute_robust_accuracy(clean_model, test_loader, attack=atk_clean, device=device, max_batches=args.max_batches)
        res_adv = compute_robust_accuracy(adv_model, test_loader, attack=atk_adv, device=device, max_batches=args.max_batches)

        c_acc = res_clean["robust_accuracy"] * 100
        a_acc = res_adv["robust_accuracy"] * 100
        delta = a_acc - c_acc

        print(f"{name:<32} | {c_acc:>14.2f}% | {a_acc:>14.2f}% | {delta:>+6.2f}%")

        results.append({
            "attack": name,
            "clean_model_robust_acc": float(c_acc),
            "adv_model_robust_acc": float(a_acc),
            "gain": float(delta),
        })

    print("=" * 80 + "\n")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Benchmark results saved to {args.output}")

if __name__ == "__main__":
    main()
