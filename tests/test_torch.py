import unittest

import torch
import torch.nn as nn


class TestTorch(unittest.TestCase):

    def test_norm(self):
        N, H, W, C = 2, 2, 3, 2
        input = torch.arange(24, dtype=torch.float32).reshape(2, 2, 3, 2)
        print(f"input:\n{input}")
        layer_norm = nn.LayerNorm([H, W, C], elementwise_affine=False)
        output = layer_norm(input)
        print(f"output:\n{output}")

        mean = torch.mean(input, range(1, input.ndim), True)
        std = torch.std(input, range(1, input.ndim), keepdim=True, unbiased=False)
        output = (input - mean) / (std + 1e-5)
        print(f"output:\n{output}")