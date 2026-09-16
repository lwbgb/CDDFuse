import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to("cpu")
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to("cpu")
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
    

def complex_decoupling_loss(feat_V, feat_I):
    # 将 64 维特征拆分为伪复数的实部(32)和虚部(32)
    real_V, imag_V = feat_V.chunk(2, dim=1)
    real_I, imag_I = feat_I.chunk(2, dim=1)
    
    # 计算振幅 (能量先验 -> 共享特征 -> 应当高度相关)
    amp_V = torch.sqrt(real_V**2 + imag_V**2 + 1e-6)
    amp_I = torch.sqrt(real_I**2 + imag_I**2 + 1e-6)
    cc_amp = cc(amp_V, amp_I) # 优化目标：最大化 cc_amp
    
    # 计算相位 (结构先验 -> 特异性特征 -> 应当正交/不相关)
    phase_V = torch.atan2(imag_V, real_V + 1e-6)
    phase_I = torch.atan2(imag_I, real_I + 1e-6)
    cc_phase = cc(phase_V, phase_I) # 优化目标：最小化 cc_phase
    
    return cc_amp, cc_phase


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()