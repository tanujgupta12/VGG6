# CS6886W - Systems Engineering for Deep Learning: Assignment 1
Department of Computer Science and Engineering, IIT Madras

This repository contains the complete modular implementation for training, evaluating, and running W&B hyperparameter sweeps for **VGG6 on CIFAR-10**.

## Project Structure
```text
cs6886w-vgg6-cifar10/
├── README.md
├── requirements.txt
├── sweep.yaml
├── main.py
├── evaluate.py
├── checkpoints/
│   └── (saved model weights: best_vgg6_cifar10.pth)
└── src/
    ├── __init__.py
    ├── model.py
    ├── dataset.py
    └── utils.py
```

## Setup & Installation

### Environment Setup
```bash
conda create -n cs6886w python=3.10 -y
conda activate cs6886w
pip install -r requirements.txt
```

## Question 1: Baseline Reproduction
To train the baseline VGG6 model with the best configuration (GELU, Nesterov-SGD, lr=0.05, batch_size=128, epochs=100, seed=42):
```bash
python main.py --activation gelu --optimizer nesterov --lr 0.05 --momentum 0.9 --batch_size 128 --epochs 100 --weight_decay 0.0005 --seed 42
```

## Evaluation on Pretrained Checkpoint
To evaluate the saved checkpoint on CIFAR-10 test set:
```bash
python evaluate.py --checkpoint checkpoints/best_vgg6_cifar10.pth --activation gelu
```

## Running W&B Sweeps (Questions 2 & 3)
To explore activations, optimizers, learning rates, and batch sizes using Weights & Biases:
```bash
wandb sweep sweep.yaml
wandb agent <SWEEP_ID>
```
