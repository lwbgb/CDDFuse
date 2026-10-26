# 数据预处理与 H5 读取

本次仅调整数据阶段 D1–D3。模型、训练入口、测试入口与已有训练 H5 保持原样。

## 输入与运行

在 Ubuntu-22.04 的项目目录、现有 pytorch 环境中运行：

```bash
conda activate pytorch
cd /home/sayuri/workspace/CDDFuse-main
python data_process.py root=/实际路径/datasets output_dir=data/rebuilt
```

输入目录结构：

```text
datasets/
└── MSRS_train/
    ├── ir/a.png
    └── vi/a.png
```

`root` 指向包含 `MSRS_train` 的父目录，`name` 默认 `MSRS_train`。支持子目录，IR/VIS 的相对路径和文件名必须完全一致。
相对输入、输出路径均以项目目录为基准，不依赖启动命令时的工作目录。
参数由 `configs/dataset.yaml` 提供，也可像上例一样通过 Hydra 的 `key=value` 覆盖。

IR 必须为8bit单通道原图，VIS必须为8bit RGB原图，配对空间尺寸一致，且两边均不小于crop_size。尺寸不符、16bit、灰度VIS等不会被静默缩放或转换。
目录中的非图片和 `:Zone.Identifier` 等附属文件被忽略；`.npy`不作为原始图像输入。

## 保持的 CDDFuse 规则

- 先转float32并除255，再按原运算顺序计算 `0.299R + 0.587G + 0.114B`。
- 默认128×128固定裁块、stride200，不补边、不缩放、不加入增强。
- 按源文件相对路径字典序遍历；每张图先遍历行位置，再遍历列位置。
- 使用10/90百分位、阈值0.1；IR或VIS任一为低对比度则剔除这一对patch，全黑块剔除。
- H5仍为`ir_patchs`和`vis_patchs`两个float32数组，形状`[N,1,H,W]`。

裁块使用NumPy窗口视图，替代逐像素Python循环。回归测试要求像素和patch顺序与独立旧规则参考逐元素一致。
新生成H5的chunk为一张patch，便于随机读取；这是存储布局优化，不改变数值，也不重写既有H5。

## 已有数据保护

默认输出为`data/MSRS_train_imgsize_128_stride_200.h5`。已有同名文件时直接报错，绝不默认覆盖。
仅在明确要重建时使用：

```bash
python data_process.py root=/实际路径/datasets overwrite=true
```

程序先向输出目录的临时文件写入全部patch，关闭后重新检查数量、形状、float32类型、有限值和范围，再发布最终文件。
未启用覆盖时用原子硬链接发布，即使中途出现同名文件也不会覆盖；显式覆盖时用原子替换。
图像检查、写入或最终检查失败时清理临时文件，已有文件保持不变。

当前正式H5未重建，仍为此前验证过的6487对patch。预处理依赖的原图目录当前未提供，所以本轮验证使用临时合成图像，不声称已完成全量原图重建。

## H5内的来源记录

新生成的H5还包含一个字符串数据集`preprocess_manifest`，记录版本、裁块/筛选/归一化参数、源图相对文件名及SHA256、尺寸、各源图保留/剔除数量与总计数。
元数据与patch放在同一个H5中一起发布，不生成需要分别提交的外部清单。
现有读取器只访问两组patch数组，忽略额外元数据；旧的无清单数组H5也可继续读取。

```python
import h5py
import json

with h5py.File("data/rebuilt/MSRS_train_imgsize_128_stride_200.h5", "r") as f:
    manifest = json.loads(f["preprocess_manifest"].asstr()[()])
```

## H5读取与随机流

读取器检查非空、配对`[N,1,H,W]`、float32，并在取样时拒绝非有限值或越界值；仍返回`(IR, VIS)`。
每个worker独立打开HDF5；fork检测PID变化，spawn序列化时不携带句柄。数据集提供`close()`用于主动释放文件。

采样随机数使用配置`seed`，worker初始化另用`seed+1`，避免worker建立/重建影响下一epoch的shuffle。
本轮验证0和2个worker在两个epoch中的采样顺序一致，每个epoch样本均完整覆盖，epoch之间仍正常重新打乱。
**这一调整可能改变与旧版共享generator运行相比的样本顺序，即使seed相同也不应直接比较旧损失轨迹。** 数据内容、IR/VIS顺序和模型公式没有改变；旧记录与检查点不会被改写，也不因此获得精确续训保证。

## 验证

```bash
python -m unittest discover -s tests -p test_preprocessing.py -v
```

13项检查覆盖patch像素/顺序、筛选、严格配对、非法图像、已有文件保护、半途失败、校验/发布失败、并发同名文件保护、CLI路径、H5读取/句柄序列化及两epoch多worker随机流。
