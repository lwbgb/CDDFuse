from omegaconf import DictConfig
import torch.utils.data as Data
from torch.utils.data import DataLoader
import h5py
import numpy as np
import torch

from schemas.train_config import TrainConfig

class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        h5f.close()
        return torch.Tensor(VIS), torch.Tensor(IR)
    

def get_loader(opt: DictConfig, dataset: Data.Dataset) -> DataLoader:
    in_order = not opt.num_threads > 0
    data_loader = DataLoader(
        dataset, opt.batch_size, opt.shuffle, num_workers=opt.num_threads, drop_last=opt.drop_last, pin_memory=torch.cuda.is_available(), persistent_workers=True, in_order=in_order)
    return data_loader