from pathlib import Path

from models.cddfuse_model import CDDFuseModel
import os
import numpy as np
from schemas.base_config import BaseConfig
from schemas.custom_dataset import CustomImageDataset
from schemas.test_config import TestConfig
from utils import create_dataset
from utils.Evaluator import Evaluator
import torch
from utils.dataset import get_loader
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
from utils.logger_initializer import logger, init_logger
from hydra import initialize, compose
from torchvision.transforms import v2

init_logger("test.log")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    for dataset_name in ["MSRS"]:
        with initialize(version_base=None, config_path="./configs"):
                opt: BaseConfig | TestConfig = compose(config_name="config", 
                                                        overrides=["+mode@_global_=test"])

        model_name = opt.model
        test_folder = Path(opt.root) / dataset_name
        test_out_folder = Path(opt.output_dir) / dataset_name

        transform_PIL_to_tensor = v2.Compose([v2.ToImage(), 
                                              v2.ToDtype(torch.float32, scale=True)])
        dataset = CustomImageDataset(root=test_folder, ir_dir="ir", vis_dir="vi", transform=transform_PIL_to_tensor)
        print(f"Dataset size of {dataset_name}: {len(dataset)}")
        dataloader = get_loader(opt, dataset)
        model: CDDFuseModel = CDDFuseModel(opt)
        model.setup()
        model.eval()

        for i, data in enumerate(dataloader):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            # visuals = model.get_current_visuals()  # get image results

            data_Fuse = (model.data_Fuse - torch.min(model.data_Fuse)) / (torch.max(model.data_Fuse) - torch.min(model.data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            img_save(fi, model.ir_img_name.split('.')[0], test_out_folder)

        ori_img_folder = str(test_folder)
        eval_folder = str(test_out_folder)
        all_eval_imgs = sorted(os.listdir(os.path.join(ori_img_folder, "ir")))

        metric_result = np.zeros((8))
        for img_name in all_eval_imgs:
            ir = image_read_cv2(os.path.join(ori_img_folder, "ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder, "vi", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0] + ".png"), 'GRAY')
            
            metric_result += np.array([
                Evaluator.EN(fi), Evaluator.SD(fi),
                Evaluator.SF(fi), Evaluator.MI(fi, ir, vi),
                Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi),
                Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
            ])

        metric_result /= len(all_eval_imgs)

        logger.info("=" * 80)
        logger.info(f"Test model: {model_name} on {dataset_name} :")
        logger.info("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
        logger.info(
            model_name + '\t' +
            str(np.round(metric_result[0], 2)) + '\t' +
            str(np.round(metric_result[1], 2)) + '\t' +
            str(np.round(metric_result[2], 2)) + '\t' +
            str(np.round(metric_result[3], 2)) + '\t' +
            str(np.round(metric_result[4], 2)) + '\t' +
            str(np.round(metric_result[5], 2)) + '\t' +
            str(np.round(metric_result[6], 2)) + '\t' +
            str(np.round(metric_result[7], 2))
        )
        logger.info("=" * 80)
