"""E1: raw [0, 1] fusion output, equally weighted source structure."""
import torch
from torch import nn
from kornia.metrics import ssim


class DualSourceSSIMLoss(nn.Module):
    """Return mean source 1-SSIM, infrared error, visible error."""

    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("SSIM window_size must be positive and odd")
        self.window_size = window_size

    def forward(
        self, fused: torch.Tensor, infrared: torch.Tensor, visible: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if fused.ndim != 4 or fused.shape != infrared.shape or fused.shape != visible.shape:
            raise ValueError("SSIM inputs must have identical [B, C, H, W] shapes")
        with torch.amp.autocast(device_type=fused.device.type, enabled=False):
            with torch.backends.cudnn.flags(allow_tf32=False):
                fused, infrared, visible = fused.float(), infrared.float(), visible.float()
                # No output normalization or clipping of 1-SSIM; range is [0, 1].
                loss_ir = 1.0 - ssim(fused, infrared, self.window_size, max_val=1.0).mean()
                loss_vis = 1.0 - ssim(fused, visible, self.window_size, max_val=1.0).mean()
                return 0.5 * (loss_ir + loss_vis), loss_ir, loss_vis
