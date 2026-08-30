import os

import torch

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print("逻辑线程数:", os.cpu_count())