import unittest
from utils.logger_initializer import logger


class TestConf(unittest.TestCase):
    
    def test_01(self):
        logger.info("test logger info")
        ...