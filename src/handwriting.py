"""Segment a single line of separated digits and classify with the saved MLP."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import torch

from src.model import MLP
from src.data import MNIST_MEAN, MNIST_STD


def corrected_grayscale(image_path):
    """Remove slow illumination changes; return dark strokes on white paper."""
    with Image.open(image_path) as source:
        gray = ImageOps.exif_transpose(source).convert('L')
    array = np.asarray(gray, dtype=np.float32)
    if np.median(array) < 127:
        gray = ImageOps.invert(gray)
        array = 255 - array
    background = np.asarray(
        gray.filter(ImageFilter.GaussianBlur(max(gray.size) / 40)), dtype=np.float32
    )
    ink = np.maximum(background - array, 0)
    foreground = ink[ink >= 8]
    if not foreground.size:
        raise ValueError('No ink detected. Use a clear, tightly cropped photo.')
    # Estimate stroke contrast from foreground pixels, independent of predictions.
    contrast = float(np.percentile(foreground, 99))
    ink = np.clip(ink * (255 / contrast), 0, 255).astype(np.uint8)
    return Image.fromarray(255 - ink)


def clean_components(mask, min_area):
    """Remove small 8-connected specks and artifacts touching the frame."""
    remaining = mask.copy()
    cleaned = np.zeros_like(mask)
    height, width = mask.shape
    for y, x in zip(*np.where(mask)):
        if not remaining[y, x]:
            continue
        remaining[y, x] = False
        stack = [(int(y), int(x))]
        pixels = []
        touches_edge = False
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            touches_edge |= cy in (0, height - 1) or cx in (0, width - 1)
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if remaining[ny, nx]:
                        remaining[ny, nx] = False
                        stack.append((ny, nx))
        if len(pixels) >= min_area and not touches_edge:
            ys, xs = zip(*pixels)
            cleaned[ys, xs] = True
    return cleaned


def split_digits(image_path, threshold=110, min_area=20):
    """Find left-to-right ink runs after correcting the paper background.

    Digits must be separated by blank columns. The returned grayscale image has
    corrected polarity and illumination and must also be used for normalization.
    """
    if not 1 <= threshold <= 254 or min_area < 1:
        raise ValueError('threshold must be 1..254 and min-area must be positive')
    gray = corrected_grayscale(image_path)
    ink = 255 - np.asarray(gray, dtype=np.uint8)
    mask = clean_components(ink > threshold, min_area)
    occupied = np.flatnonzero(mask.any(axis=0))
    if not occupied.size:
        raise ValueError('No digit-sized regions found. Check the photo or --threshold/--min-area.')
    runs = np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1)
    boxes = []
    for run in runs:
        x0, x1 = int(run[0]), int(run[-1]) + 1
        ys = np.flatnonzero(mask[:, x0:x1].any(axis=1))
        boxes.append((x0, int(ys[0]), x1, int(ys[-1]) + 1))
    # Keep antialiasing near retained strokes, suppress discarded paper noise.
    support = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(
        ImageFilter.MaxFilter(3)
    )) > 0
    return Image.fromarray(255 - np.where(support, ink, 0).astype(np.uint8)), boxes


def mnist_digit(image, bbox, mean=MNIST_MEAN, std=MNIST_STD):
    """Preserve aspect ratio, fit to 20px, and center by ink mass in 28x28."""
    if std <= 0:
        raise ValueError('Normalization std must be positive')
    patch = ImageOps.invert(image.crop(bbox))
    w, h = patch.size
    if w == 0 or h == 0:
        raise ValueError('Empty character')
    scale = 20 / max(w, h)
    resized = patch.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS
    )
    canvas = Image.new('L', (28, 28), 0)
    canvas.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))
    ink = np.asarray(canvas, dtype=np.float32)
    mass = float(ink.sum())
    if mass == 0:
        raise ValueError('Empty character')
    ys, xs = np.indices(ink.shape)
    dx = round(13.5 - float((xs * ink).sum() / mass))
    dy = round(13.5 - float((ys * ink).sum() / mass))
    centered = Image.new('L', (28, 28), 0)
    centered.paste(canvas, (dx, dy))
    raw = np.asarray(centered, dtype=np.float32) / 255.0
    normalized = (raw - mean) / std
    return centered, torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)


def compare_sequence(predicted, expected):
    """Score positions only with equal lengths; never guess an alignment."""
    if not expected or any(digit not in '0123456789' for digit in expected):
        raise ValueError('Expected ID must contain ASCII digits only')
    matching_lengths = len(predicted) == len(expected)
    correct = sum(a == b for a, b in zip(predicted, expected)) if matching_lengths else None
    return {
        'expected_id': expected,
        'expected_count': len(expected),
        'digit_count_matches': matching_lengths,
        'exact_match': predicted == expected,
        'correct_digits': correct,
        'digit_accuracy_pct': 100 * correct / len(expected) if matching_lengths else None,
        'mismatches': [
            {'position': i, 'expected_digit': a, 'predicted_digit': b}
            for i, (a, b) in enumerate(zip(expected, predicted), start=1) if a != b
        ] if matching_lengths else None,
    }


def predict(args):
    checkpoint_file = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_file is None:
        summary = json.loads(Path('results/summary.json').read_text(encoding='utf8'))
        checkpoint_file = Path('models') / (summary['selected_model'] + '.pt')
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
    model = MLP(checkpoint['hidden_sizes'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    mean, std = checkpoint.get('mean', MNIST_MEAN), checkpoint.get('std', MNIST_STD)
    # Ground truth is deliberately not passed to segmentation or the model.
    gray, boxes = split_digits(args.image, args.threshold, args.min_area)
    expected = getattr(args, 'expected', None)
    if expected is not None and (not expected or any(d not in '0123456789' for d in expected)):
        raise ValueError('Expected ID must contain ASCII digits only')
    output_dir = Path(getattr(args, 'output_dir', 'results/handwriting'))
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(args.image) as source:
        annotated = ImageOps.exif_transpose(source).convert('RGB')
    draw = ImageDraw.Draw(annotated)
    rows, digit_images = [], []
    for i, bbox in enumerate(boxes, start=1):
        digit_image, tensor = mnist_digit(gray, bbox, mean, std)
        digit_image.save(output_dir / f'digit_{i}.png')
        digit_images.append(digit_image)
        with torch.inference_mode():
            probs = torch.softmax(model(tensor), dim=1).squeeze(0)
        predicted = int(probs.argmax().item())
        confidence = float(probs[predicted].item())
        row = {'position': i, 'predicted_digit': predicted,
               'confidence': confidence, 'bounding_box': list(bbox)}
        if expected is not None and len(expected) == len(boxes):
            row.update(expected_digit=int(expected[i - 1]), correct=str(predicted) == expected[i - 1])
        rows.append(row)
        draw.rectangle(bbox, outline='red', width=2)
        draw.text((bbox[0], max(0, bbox[1] - 15)), f'{i}: {predicted}', fill='red')
    with (output_dir / 'predictions.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    sequence = ''.join(str(row['predicted_digit']) for row in rows)
    report = {
        'image': Path(args.image).as_posix(), 'checkpoint': checkpoint_file.as_posix(),
        'experiment': checkpoint.get('experiment'), 'best_epoch': checkpoint.get('best_epoch'),
        'preprocessing': {'threshold': args.threshold, 'min_area': args.min_area,
                          'background_blur_radius': max(gray.size) / 40,
                          'mean': mean, 'std': std, 'centering': 'center_of_mass'},
        'predicted_id': sequence, 'detected_count': len(boxes), 'predictions': rows,
    }
    if expected is not None:
        report.update(compare_sequence(sequence, expected))
    (output_dir / 'summary.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf8'
    )
    gray.save(output_dir / 'preprocessed.png')
    annotated.save(output_dir / 'segmentation.png')
    strip = Image.new('L', (len(digit_images) * 36, 36), 0)
    for i, digit_image in enumerate(digit_images):
        strip.paste(digit_image, (i * 36 + 4, 4))
    strip.resize((strip.width * 3, strip.height * 3), Image.Resampling.NEAREST).save(
        output_dir / 'digits.png'
    )
    print('Predicted sequence:', sequence)
    print('Individual probabilities:', ', '.join(
        f"{r['predicted_digit']} ({r['confidence']:.1%})" for r in rows
    ))
    print(f'Detected {len(boxes)} separated digits. Outputs: {output_dir}')
    if expected is not None:
        print('Expected sequence:', expected)
        if report['digit_count_matches']:
            print(f"Correct digits: {report['correct_digits']}/{len(expected)} "
                  f"({report['digit_accuracy_pct']:.2f}%) | Exact match: {report['exact_match']}")
        else:
            print('Digit counts differ. Fix segmentation before scoring individual digits.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='Path to your handwritten student ID photo')
    parser.add_argument('--checkpoint', help='Model .pt; default: selected model in results/summary.json')
    parser.add_argument('--threshold', type=int, default=110, help='Corrected ink threshold, 1..254')
    parser.add_argument('--min-area', type=int, default=20)
    parser.add_argument('--expected', help='True ID used only to score predictions')
    parser.add_argument('--output-dir', default='results/handwriting')
    arguments = parser.parse_args()
    try:
        predict(arguments)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f'Error: {error}\n')

