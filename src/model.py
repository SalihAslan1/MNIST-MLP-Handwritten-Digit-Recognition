import torch.nn as nn

class MLP(nn.Module):
    """Fully-connected MNIST classifier; no convolutional layers."""
    def __init__(self, hidden_sizes=(256, 128)):
        super().__init__()
        layers = [nn.Flatten()]
        prev = 784
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        layers.append(nn.Linear(prev, 10))  # raw logits for CrossEntropyLoss
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
