"""Boundary tests for the new optional SFT collator; no model weights required."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # Windows DLL order.
import torch
from dataset.batch_collator import DynamicRightPadding


class CollatorTests(unittest.TestCase):
    def test_preserves_unsupervised_context_and_supervised_eos(self):
        x = torch.tensor([1, 7, 0, 8, 2, 9, 0, 0, 0, 0, 0, 0])
        y = torch.tensor([-100, -100, -100, 8, 2, -100, -100, -100, -100, -100, -100, -100])
        a, b = DynamicRightPadding(multiple=1)([(x, y)])
        self.assertEqual(a.shape, (1, 6))
        self.assertTrue(torch.equal(a[0], x[:6]))
        self.assertTrue(torch.equal(b[0], y[:6]))

    def test_alignment_and_batch_maximum(self):
        a = torch.tensor([1] * 9 + [0] * 7)
        b = torch.tensor([1] * 3 + [0] * 13)
        labels = torch.full((16,), -100)
        x, _ = DynamicRightPadding(multiple=8)([(a, labels), (b, labels)])
        self.assertEqual(x.shape, (2, 16))
        x, _ = DynamicRightPadding(multiple=1)([(a, labels), (b, labels)])
        self.assertEqual(x.shape, (2, 9))

    def test_supervised_pad_not_removed(self):
        x = torch.tensor([1, 2, 0, 0, 0, 0])
        y = torch.tensor([-100, 2, -100, 0, -100, -100])
        a, b = DynamicRightPadding(multiple=1)([(x, y)])
        self.assertEqual(a.shape, (1, 4))
        self.assertEqual(b[0, -1].item(), 0)

    def test_authoritative_length_preserves_trailing_real_pad_id(self):
        x = torch.tensor([1, 2, 0, 0, 0, 0])
        y = torch.tensor([-100, 2, -100, -100, -100, -100])
        a, _ = DynamicRightPadding(multiple=1)([(x, y, 4)])
        self.assertEqual(a.shape, (1, 4))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): DynamicRightPadding(multiple=0)
        with self.assertRaises(ValueError): DynamicRightPadding()([])
        x = torch.ones(4, dtype=torch.long)
        with self.assertRaises(ValueError): DynamicRightPadding()([(x, x, 9)])
        with self.assertRaises(ValueError): DynamicRightPadding()([(x, x), (x, x, 4)])


if __name__ == '__main__':
    unittest.main()
