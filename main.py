import argparse
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import get_dataloaders
from src.model import VGG6
from src.utils import set_seed, get_optimizer

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / total, 100.0 * correct / total

def evaluate_loader(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss / total, 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser(description="Train VGG6 on CIFAR-10")
    parser.add_argument('--activation', type=str, default='gelu', choices=['relu', 'gelu', 'silu', 'tanh', 'sigmoid'])
    parser.add_argument('--optimizer', type=str, default='nesterov', choices=['sgd', 'nesterov', 'adam', 'nadam', 'rmsprop', 'adagrad'])
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    wb_run = None
    if args.use_wandb:
        import wandb
        wb_run = wandb.init(project="vgg6-cifar10-assignment1", config=vars(args))

    # Q1(a): Data preparation
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Q1(b): Model & Optimizer initialization
    model = VGG6(activation=args.activation, use_bn=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args.optimizer, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)

    best_val_acc = 0.0
    checkpoint_path = os.path.join(args.save_dir, 'best_vgg6_cifar10.pth')

    print(f"Starting training: {args.epochs} epochs | Act: {args.activation} | Opt: {args.optimizer} | LR: {args.lr}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_loader(model, test_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if wb_run is not None:
            wb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]['lr']
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': vars(args)
            }, checkpoint_path)

    print(f"Training completed. Best Val Acc: {best_val_acc:.2f}%. Saved to {checkpoint_path}")
    if wb_run is not None:
        wb_run.finish()

if __name__ == '__main__':
    main()
