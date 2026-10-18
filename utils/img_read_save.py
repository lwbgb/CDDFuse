import numpy as np
import cv2
import os
from skimage.io import imsave

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    imsave(os.path.join(savepath, "{}.png".format(imagename)),image.astype(np.uint8))  # 更改：转换保存的图片格式为 int 类型
    

def restore_color(fused_gray_path, visible_path, output_path):
    # 使用原始彩色可见光图，而非此前保存的灰度版本。
    visible = cv2.imread(visible_path, cv2.IMREAD_COLOR)
    fused = cv2.imread(fused_gray_path, cv2.IMREAD_GRAYSCALE)

    if visible is None or fused is None:
        raise FileNotFoundError("无法读取可见光图或融合图")
    if visible.shape[:2] != fused.shape:
        raise ValueError("可见光图与融合图的空间尺寸必须一致")

    # OpenCV 的通道顺序是 Y、Cr、Cb。
    ycrcb = cv2.cvtColor(
        visible.astype(np.float32) / 255.0,
        cv2.COLOR_BGR2YCrCb,
    )
    ycrcb[..., 0] = fused.astype(np.float32) / 255.0

    color = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    color = np.round(np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)

    if not cv2.imwrite(output_path, color):
        raise OSError(f"保存失败：{output_path}")