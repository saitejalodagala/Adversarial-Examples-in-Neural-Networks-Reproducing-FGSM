import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

from adv_studio.models import get_model, SimpleCNN, LeNet5, MiniResNet18, MLP
from adv_studio.data import get_mnist_loaders
from adv_studio.attacks import get_attack, FGSMAttack, PGDAttack
from adv_studio.defenses import AdversarialTrainer
from adv_studio.evaluation import compute_robust_accuracy, compute_distortion_metrics


def cli_train(args):
    """Train a clean or robust model."""
    print(f"[*] Training model: {args.model} | Epochs: {args.epochs} | Batch size: {args.batch_size}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[*] Hardware device: {device}")

    train_loader, test_loader = get_mnist_loaders(data_dir=args.data_dir, batch_size=args.batch_size)
    model = get_model(name=args.model, device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.adv_train:
        print(f"[*] Enabling Adversarial Training (Type: {args.adv_attack}, Epsilon: {args.epsilon})")
        trainer = AdversarialTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            attack_type=args.adv_attack,
            epsilon=args.epsilon,
            device=device,
        )
        for ep in range(1, args.epochs + 1):
            loss, clean_acc, adv_acc = trainer.train_epoch(train_loader)
            print(f"Epoch [{ep}/{args.epochs}] Loss: {loss:.4f} | Clean Acc: {clean_acc*100:.2f}% | Adv Acc: {adv_acc*100:.2f}%")
    else:
        for ep in range(1, args.epochs + 1):
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
            print(f"Epoch [{ep}/{args.epochs}] Train Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.2f}%")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"[+] Model checkpoint saved to: {args.output}")


def cli_eval(args):
    """Evaluate model robustness."""
    print(f"[*] Evaluating model: {args.model} | Checkpoint: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    _, test_loader = get_mnist_loaders(data_dir=args.data_dir, batch_size=args.batch_size)
    model = get_model(name=args.model, pretrained_path=args.checkpoint, device=device)

    attack = None
    if args.attack:
        attack = get_attack(name=args.attack, model=model, epsilon=args.epsilon, device=device)
        print(f"[*] Attack: {args.attack} | Epsilon: {args.epsilon}")

    metrics = compute_robust_accuracy(model=model, loader=test_loader, attack=attack, device=device, max_batches=args.max_batches)
    print("=" * 45)
    print(f" Clean Accuracy:        {metrics['clean_accuracy']*100:.2f}%")
    print(f" Robust Accuracy:       {metrics['robust_accuracy']*100:.2f}%")
    print(f" Attack Success Rate:   {metrics['attack_success_rate']*100:.2f}%")
    print(f" Mean Confidence:       {metrics['mean_adv_confidence']*100:.2f}%")
    print("=" * 45)


def cli_serve(args):
    """Launch the interactive web studio."""
    import uvicorn
    from adv_studio.web.server import app

    print(f"[+] Launching Adversarial Studio at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        prog="adv-studio",
        description="Adversarial Studio: Research & Reproduction Platform for Neural Network Adversaries",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Train subcommand
    train_parser = subparsers.add_parser("train", help="Train clean or adversarially robust models")
    train_parser.add_argument("--model", type=str, default="simple_cnn", help="Model architecture")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--data-dir", type=str, default="./DATA", help="Path to MNIST data")
    train_parser.add_argument("--adv-train", action="store_true", help="Enable adversarial training")
    train_parser.add_argument("--adv-attack", type=str, default="fgsm", help="Attack for adv training")
    train_parser.add_argument("--epsilon", type=float, default=0.2, help="Epsilon perturbation budget")
    train_parser.add_argument("--output", type=str, default="checkpoints/model.pth", help="Output path")
    train_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")

    # 2. Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate model robustness")
    eval_parser.add_argument("--model", type=str, default="simple_cnn", help="Model architecture")
    eval_parser.add_argument("--checkpoint", type=str, default="checkpoints/mnist_cnn.pth", help="Checkpoint")
    eval_parser.add_argument("--attack", type=str, default="fgsm", help="Attack algorithm")
    eval_parser.add_argument("--epsilon", type=float, default=0.2, help="Epsilon")
    eval_parser.add_argument("--data-dir", type=str, default="./DATA", help="Data directory")
    eval_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    eval_parser.add_argument("--max-batches", type=int, default=10, help="Max test batches to evaluate")
    eval_parser.add_argument("--cpu", action="store_true", help="Force CPU")

    # 3. Serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Launch the real-time Interactive Web Studio")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()
    if args.command == "train":
        cli_train(args)
    elif args.command == "eval":
        cli_eval(args)
    elif args.command == "serve":
        cli_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
