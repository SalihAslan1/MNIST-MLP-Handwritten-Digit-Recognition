import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data import build_loaders
from src.model import MLP

EXPERIMENTS = {
    'baseline': {'hidden_sizes': [256, 128], 'lr': 0.001},
    'small': {'hidden_sizes': [128, 64], 'lr': 0.001},
    'higher_lr': {'hidden_sizes': [256, 128], 'lr': 0.01},
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def one_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            scores = model(images)
            loss = criterion(scores, targets)
            if training:
                loss.backward()
                optimizer.step()
            n = targets.size(0)
            total_loss += loss.item() * n
            correct += (scores.argmax(dim=1) == targets).sum().item()
            total += n
    return total_loss / total, 100 * correct / total

def draw_curves(history, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    epochs = list(range(1, len(history) + 1))
    for key, label in [('train_loss','Training'), ('val_loss','Validation')]:
        axs[0].plot(epochs, [r[key] for r in history], marker='o', label=label)
    for key, label in [('train_acc','Training'), ('val_acc','Validation')]:
        axs[1].plot(epochs, [r[key] for r in history], marker='o', label=label)
    axs[0].set(title='Cross-entropy loss', xlabel='Epoch', ylabel='Loss')
    axs[1].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy (%)')
    for ax in axs:
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def run(args):
    results_dir, models_dir = Path('results'), Path('models')
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | data split: 50000 train, 10000 validation, 10000 test')
    # The test loader is intentionally not used until after all model selection.
    train_loader, val_loader, test_loader = build_loaders(args.batch_size, args.seed)
    summaries = []
    for name, config in EXPERIMENTS.items():
        seed_everything(args.seed)
        model = MLP(config['hidden_sizes']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.CrossEntropyLoss()
        history = []
        best_val_loss = float('inf')
        best_epoch = 0
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = one_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc = one_epoch(model, val_loader, criterion, device)
            history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                                val_loss=val_loss, val_acc=val_acc))
            print(f'{name}: epoch {epoch}/{args.epochs} | train loss {train_loss:.4f}, '
                  f'train acc {train_acc:.2f}% | val loss {val_loss:.4f}, val acc {val_acc:.2f}%')
            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch
                torch.save({'state_dict': model.state_dict(), 'hidden_sizes': config['hidden_sizes'],
                            'mean': 0.1307, 'std': 0.3081,
                            'experiment': name, 'best_epoch': epoch}, models_dir / f'{name}.pt')
        seconds = time.perf_counter() - started
        with (results_dir / f'{name}_metrics.csv').open('w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        draw_curves(history, results_dir / f'{name}_curves.png')
        best_row = history[best_epoch - 1]
        summaries.append({'experiment': name, 'hidden_sizes': config['hidden_sizes'],
                          'learning_rate': config['lr'], 'batch_size': args.batch_size,
                          'epochs_requested': args.epochs, 'best_epoch': best_epoch,
                          'train_loss': best_row['train_loss'], 'val_loss': best_row['val_loss'],
                          'train_accuracy_pct': best_row['train_acc'],
                          'val_accuracy_pct': best_row['val_acc'], 'training_seconds': seconds})
    # Select only using validation loss, never test score.
    best = min(summaries, key=lambda row: row['val_loss'])
    checkpoint = torch.load(models_dir / f"{best['experiment']}.pt", map_location=device, weights_only=True)
    model = MLP(checkpoint['hidden_sizes']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    test_loss, test_acc = one_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    best['test_loss'] = test_loss
    best['test_accuracy_pct'] = test_acc
    with (results_dir / 'summary.json').open('w', encoding='utf8') as f:
        json.dump({'seed': args.seed, 'data_split': {'train': 50000, 'validation': 10000,
                   'test': 10000}, 'model_selection_metric': 'validation_loss',
                   'selected_model': best['experiment'], 'experiments': summaries}, f,
                  indent=2, ensure_ascii=False)
    print(f"\nSelected: {best['experiment']} (minimum validation loss). ")
    print(f'Final test accuracy: {test_acc:.2f}% | test loss: {test_loss:.4f}')
    print('Outputs saved under results/ and models/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.batch_size < 1:
        parser.error('--epochs and --batch-size must be positive')
    run(arguments)
