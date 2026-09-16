import os
from pathlib import Path
import pandas as pd
import torch
from torchvision.io import decode_image
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2

class CustomImageDataset(Dataset):
    def __init__(self, root: str | Path, ir_dir: str, vis_dir: str, transform=None):
        self._root = Path(root)
        self._ir_path = self._root / ir_dir
        self._vis_path = self._root / vis_dir
        self._ir_images = sorted(os.listdir(self._ir_path))
        self._vis_images = sorted(os.listdir(self._vis_path))
        assert len(self._ir_images) == len(self._vis_images)
        self._items = [(ir, vis) for ir, vis in zip(self._ir_images, self._vis_images)]
        self.transform = transform

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        ir_img_name, vis_img_name = self._items[idx]
        ir_img_path = self._ir_path / ir_img_name
        vis_img_path = self._vis_path / vis_img_name
        ir_image = Image.open(ir_img_path)
        vis_image = Image.open(vis_img_path)
        if self.transform:
            ir_image = self.transform(ir_image)
            vis_image = self.transform(vis_image)
        return (ir_image, vis_image)


if __name__ == "__main__":
    transform_PIL_to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    dataset = CustomImageDataset(root="test_img/MSRS", ir_dir="ir", vis_dir="vi", transform=transform_PIL_to_tensor)
    print(f"Dataset size: {len(dataset)}")
    for i in range(len(dataset)):
        ir_image, vis_image = dataset[i]
        print(f"Sample {i}: IR image size: {ir_image.shape}, VIS image size: {vis_image.shape}")
        print(f"Type of IR image: {type(ir_image)}, Type of VIS image: {type(vis_image)}")
        break