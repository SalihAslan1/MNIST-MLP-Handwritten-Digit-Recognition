"""Train and compare all 3 MLP configurations."""
from src.train import run
import argparse

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1:
        p.error('epochs and batch-size must be positive')
    run(a)
