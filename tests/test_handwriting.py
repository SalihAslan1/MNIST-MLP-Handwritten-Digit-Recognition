"""Regression checks for segmentation and scoring; fixtures are geometric shapes."""
import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.handwriting import compare_sequence, mnist_digit, predict, split_digits


ROOT = Path(__file__).resolve().parents[1]


class HandwritingTests(unittest.TestCase):
    def test_shadow_and_noise_do_not_merge_separate_shapes(self):
        background = np.tile(np.linspace(120, 205, 240), (100, 1))
        background[25:75, 45:50] = 30
        background[25:75, 165:170] = 40
        background[10, 110] = 0  # A speck must not become a third region.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shapes.png'
            Image.fromarray(background.astype(np.uint8)).save(path)
            _, boxes = split_digits(path)
        self.assertEqual(len(boxes), 2)
        self.assertLess(boxes[0][2], boxes[1][0])

    def test_inverted_background_has_same_preprocessing(self):
        shapes = np.full((100, 240), 255, dtype=np.uint8)
        shapes[20:80, 45:50] = 0
        shapes[20:80, 165:170] = 0
        with tempfile.TemporaryDirectory() as directory:
            light = Path(directory) / 'light.png'
            dark = Path(directory) / 'dark.png'
            image = Image.fromarray(shapes)
            image.save(light)
            ImageOps.invert(image).save(dark)
            corrected_light, light_boxes = split_digits(light)
            corrected_dark, dark_boxes = split_digits(dark)
        self.assertEqual(light_boxes, dark_boxes)
        self.assertEqual(len(light_boxes), 2)
        np.testing.assert_array_equal(corrected_light, corrected_dark)

    def test_blank_photo_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'blank.png'
            Image.new('L', (200, 100), 180).save(path)
            with self.assertRaisesRegex(ValueError, 'No ink'):
                split_digits(path)

    def test_normalization_uses_supplied_checkpoint_values(self):
        image = Image.new('L', (10, 40), 0)
        digit, tensor = mnist_digit(image, (0, 0, 10, 40), mean=0.2, std=0.5)
        self.assertEqual(digit.size, (28, 28))
        self.assertEqual(tuple(tensor.shape), (1, 1, 28, 28))
        raw = np.asarray(digit, dtype=np.float32) / 255
        np.testing.assert_allclose(tensor.numpy()[0, 0], (raw - 0.2) / 0.5)
        ys, xs = np.indices(raw.shape)
        self.assertLessEqual(abs((xs * raw).sum() / raw.sum() - 13.5), 0.5)
        self.assertLessEqual(abs((ys * raw).sum() / raw.sum() - 13.5), 0.5)

    def test_scoring_preserves_leading_zero_and_reports_wrong_positions(self):
        score = compare_sequence('0129', '0123')
        self.assertEqual(score['expected_id'], '0123')
        self.assertEqual(score['correct_digits'], 3)
        self.assertEqual(score['digit_accuracy_pct'], 75)
        self.assertEqual(score['mismatches'], [
            {'position': 4, 'expected_digit': '3', 'predicted_digit': '9'}
        ])
        self.assertFalse(score['exact_match'])

    def test_missing_digit_does_not_produce_misleading_positional_accuracy(self):
        score = compare_sequence('013', '0123')
        self.assertFalse(score['digit_count_matches'])
        self.assertIsNone(score['correct_digits'])
        self.assertIsNone(score['digit_accuracy_pct'])
        self.assertIsNone(score['mismatches'])

    def test_invalid_ground_truth_and_parameters_are_rejected(self):
        for expected in ('', '12x', '１２３'):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                compare_sequence('123', expected)
        with self.assertRaises(ValueError):
            split_digits('unused.png', threshold=255)
        with self.assertRaises(ValueError):
            split_digits('unused.png', min_area=0)

    def test_real_photo_has_eleven_ordered_regions(self):
        _, boxes = split_digits(ROOT / 'handwriting/student_id.jpeg')
        self.assertEqual(len(boxes), 11)
        self.assertTrue(all(left[2] <= right[0] for left, right in zip(boxes, boxes[1:])))
        self.assertTrue(all(x1 - x0 < 200 and y1 - y0 < 200 for x0, y0, x1, y1 in boxes))

    def test_ground_truth_does_not_change_real_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                image=str(ROOT / 'handwriting/student_id.jpeg'),
                checkpoint=str(ROOT / 'models/baseline.pt'),
                threshold=110, min_area=20, output_dir=directory, expected='23120205009',
            )
            with contextlib.redirect_stdout(io.StringIO()):
                first = predict(args)
                args.expected = '00000000000'
                second = predict(args)
            self.assertEqual(first['predicted_id'], second['predicted_id'])
            for left, right in zip(first['predictions'], second['predictions']):
                self.assertEqual(left['bounding_box'], right['bounding_box'])
                self.assertEqual(left['predicted_digit'], right['predicted_digit'])
                self.assertEqual(left['confidence'], right['confidence'])


if __name__ == '__main__':
    unittest.main()
