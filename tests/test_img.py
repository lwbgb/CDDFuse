import unittest

import torch

from schemas.custom_dataset import CustomImageDataset
from torchvision.transforms import v2


class TestImage(unittest.TestCase):
    
    def test01(self):
        transform_PIL_to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        dataset = CustomImageDataset(root="test_img/MSRS", ir_dir="ir", vis_dir="vi", transform=transform_PIL_to_tensor)
        print(f"Dataset size: {len(dataset)}")