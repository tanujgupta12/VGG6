import torch
import torch.nn as nn

def get_activation(name: str):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'silu':
        return nn.SiLU(inplace=True)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {name}")

class VGG6(nn.Module):
    """
    VGG6 architecture for CIFAR-10 (32x32x3 input).
    Consists of 5 Convolutional layers structured across 3 stages,
    followed by MaxPooling and a Fully-Connected Classifier head.
    """
    def __init__(self, activation='gelu', use_bn=True, num_classes=10):
        super(VGG6, self).__init__()
        
        def conv_block(in_f, out_f):
            layers = [nn.Conv2d(in_f, out_f, kernel_size=3, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(get_activation(activation))
            return layers

        # Feature Extractor (5 conv layers)
        self.features = nn.Sequential(
            *conv_block(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(128, 256),
            *conv_block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            get_activation(activation),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
