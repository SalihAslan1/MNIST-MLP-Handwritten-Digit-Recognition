import unittest

import torch
from src.model import MLP


def test_model_shape():
    assert MLP((256, 128))(torch.zeros(4, 1, 28, 28)).shape == (4, 10)


def test_backprop():
    model = MLP((128, 64))
    loss = torch.nn.CrossEntropyLoss()(model(torch.rand(2, 1, 28, 28)), torch.tensor([1, 4]))
    loss.backward()
    assert model.network[1].weight.grad is not None


class ModelSmokeTests(unittest.TestCase):
    def test_model_shape(self):
        test_model_shape()

    def test_backprop(self):
        test_backprop()
