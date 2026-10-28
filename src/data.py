import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])

def build_loaders(batch_size=64, seed=42, data_dir='data'):
    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=TRANSFORM)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=TRANSFORM)
    train_data, val_data = random_split(
        train_full, [50000, 10000], generator=torch.Generator().manual_seed(seed)
    )
    kwargs = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True,
                   generator=torch.Generator().manual_seed(seed), **kwargs),
        DataLoader(val_data, batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs),
    )
