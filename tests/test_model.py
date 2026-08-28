import unittest

from torch import nn, optim
import torch

from utils.checkpoint import save_epoch_checkpoint, load_epoch_checkpoint


class TestModel(unittest.TestCase):
    
    def test_load_model(self):
        ...
        