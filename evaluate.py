import argparse
import os
import torch
import torch.nn as nn
from src.model import VGG6
from src.dataset import get_dataloaders

def evaluate(checkpoint_path, activation='gelu', batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    _, test_loader = get_dataloaders(batch_size=batch_size)
    model = VGG6(activation=activation, use_bn=True).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100.0 * correct / total
    test_loss = total_loss / total
    print(f"Results for checkpoint: {checkpoint_path}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Top-1 Accuracy: {test_acc:.2f}%")
    return test_acc, test_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate VGG6 checkpoint on CIFAR-10")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_vgg6_cifar10.pth')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.activation, args.batch_size)
