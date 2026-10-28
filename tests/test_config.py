from pathlib import Path
import unittest
from utils.logger_initializer import logger


class TestConf(unittest.TestCase):
    
    def test_01(self):
        print(Path("CDDFuse_epoch_60").stem)
        ...