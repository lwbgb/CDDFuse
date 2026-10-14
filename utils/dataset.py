import os
from omegaconf import DictConfig
import torch.utils.data as Data
from torch.utils.data import DataLoader
import h5py
import numpy as np
import torch

# 1. 彻底禁用 HDF5 文件锁，防止多进程 DataLoader 读取同一 H5 时因文件锁冲突崩溃
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = str(h5file_path)
        with h5py.File(self.h5file_path, 'r') as h5f:
            self.dataset_len = h5f['ir_patchs'].shape[0]
        self.h5f = None

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        if self.h5f is None:
            self.h5f = h5py.File(self.h5file_path, 'r', swmr=True)
            
        IR = self.h5f['ir_patchs'][index]
        VIS = self.h5f['vis_patchs'][index]
        
        return torch.from_numpy(IR).float(), torch.from_numpy(VIS).float()


def get_loader(opt: DictConfig, dataset: Data.Dataset) -> DataLoader:
    num_workers = getattr(opt, 'num_threads', 0)
    
    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=opt.shuffle,
        num_workers=num_workers,
        drop_last=opt.drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )
    return data_loader