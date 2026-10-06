import os
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from hydra import initialize, compose

from schemas.base_config import BaseConfig
from schemas.dataset_config import DatasetConfig
from schemas.train_config import TrainConfig


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist
    
def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold


if __name__ == "__main__":
    with initialize(version_base=None, config_path="./configs"):
        opt: DatasetConfig = compose(config_name="dataset")
    
    img_size = opt.crop_size   #patch size
    stride = opt.stride     #patch stride

    IR_files = sorted(get_img_file(Path(opt.root) / opt.name / "ir"))
    VIS_files = sorted(get_img_file(Path(opt.root) / opt.name / "vi"))
    assert len(IR_files) == len(VIS_files)
     
    file_path = os.path.join('data', opt.name + '_imgsize_' + str(img_size) + "_stride_" + str(stride) + '.h5')
    h5f = h5py.File(file_path, 'w')
    
    # 核心修改点：创建支持动态扩容的单一数据集 (Chunked Dataset)
    # 假设图像被转为了单通道 [1, H, W]，因此数据集的总形状将是 [N, 1, img_size, img_size]
    h5_ir = h5f.create_dataset('ir_patchs', 
                               shape=(0, 1, img_size, img_size), 
                               maxshape=(None, 1, img_size, img_size), 
                               dtype=np.float32, 
                               chunks=True)
    h5_vis = h5f.create_dataset('vis_patchs', 
                                shape=(0, 1, img_size, img_size), 
                                maxshape=(None, 1, img_size, img_size), 
                                dtype=np.float32, 
                                chunks=True)
    
    train_num = 0
    for i in tqdm(range(len(IR_files))):
        I_VIS = imread(VIS_files[i]).astype(np.float32).transpose(2,0,1)/255. 
        I_VIS = rgb2y(I_VIS) 
        I_IR = imread(IR_files[i]).astype(np.float32)[None, :, :]/255.  
        
        I_IR_Patch_Group = Im2Patch(I_IR, img_size, stride)
        I_VIS_Patch_Group = Im2Patch(I_VIS, img_size, stride)  
        
        valid_IR_list = []
        valid_VIS_list = []
        
        # 收集当前原图中所有符合对比度要求的 Patch
        for ii in range(I_IR_Patch_Group.shape[-1]):
            bad_IR = is_low_contrast(I_IR_Patch_Group[0,:,:,ii])
            bad_VIS = is_low_contrast(I_VIS_Patch_Group[0,:,:,ii])
            if not (bad_IR or bad_VIS):
                avl_IR = I_IR_Patch_Group[0,:,:,ii][None, ...]
                avl_VIS = I_VIS_Patch_Group[0,:,:,ii][None, ...]
                valid_IR_list.append(avl_IR)
                valid_VIS_list.append(avl_VIS)

        # 批量将这些有效 Patch 写入 HDF5 中，减少 resize I/O 次数
        if valid_IR_list:
            batch_IR = np.stack(valid_IR_list, axis=0)   # shape: (K, 1, H, W)
            batch_VIS = np.stack(valid_VIS_list, axis=0) # shape: (K, 1, H, W)
            
            add_len = batch_IR.shape[0]
            current_len = h5_ir.shape[0]
            
            # 动态扩容
            h5_ir.resize(current_len + add_len, axis=0)
            h5_vis.resize(current_len + add_len, axis=0)
            
            # 批量赋值
            h5_ir[current_len : current_len + add_len] = batch_IR
            h5_vis[current_len : current_len + add_len] = batch_VIS
            
            train_num += add_len        

    h5f.close()
    print(f"Dataset built successfully! Total patches: {train_num}")