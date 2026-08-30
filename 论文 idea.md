

```txt
IR, VIS
   ↓
Shared Restormer Encoder
   ↓
S_ir, S_vis
   ↓
Feature Alignment Module
   ↓
Aligned S_ir, S_vis
   ↓
Base / Detail Decomposition
   ├── Base Cross-attention Fusion
   └── Detail Edge-aware INN Fusion
   ↓
Restormer Decoder
   ↓
Fused Image
```

对应改动点：
1. **SFE 后加入特征级对齐模块**；
2. **Base 分支加入 common/discrepancy cross-attention**；
3. **Detail 分支加入 edge-aware 或 gradient-aware 融合**；
4. **训练损失加入 alignment loss、frequency/gradient loss**。

该方向参考了 STFNet 中使用 deformable convolution 对轻微未配准图像进行特征对齐的思想；STFNet 明确提出用 deformable convolution-based feature align network 缓解源图像轻微错位并减少伪影。 同时也参考 ATFusion 对 cross-attention 的改进，即分别建模 common information 和 discrepancy information，避免普通注意力只偏向共同信息而忽略模态差异。

# 3. 改动一：加入 Feature Alignment Module
## 3.1 为什么要加对齐模块？
CDDFuse 默认红外和可见光图像是较好配准的，但真实数据中常存在：

- 传感器视角差异；
- 红外/可见光边缘不完全重合；
- 行人、车辆边界轻微错位；
- 数据集标定误差。

IVIF 最新综述将 **data compatibility** 作为重要方向，指出像素未对齐会影响融合模型的实际应用鲁棒性
https://arxiv.org/abs/2501.10761
https://github.com/RollingPlain/IVIF_ZOO

## 3.2 推荐插入位置
不要直接对原图做配准，优先做 **特征级对齐**：
```python
base_ir, detail_ir, shared_ir = DIDF_Encoder(data_IR)
base_vis, detail_vis, shared_vis = DIDF_Encoder(data_VIS)
aligned_ir, aligned_vis = AlignModule(shared_ir, shared_vis)
```

然后再做 base/detail 分解：
```python
base_ir = DIDF_Encoder.baseFeature(aligned_ir)
detail_ir = DIDF_Encoder.detailFeature(aligned_ir)

base_vis = DIDF_Encoder.baseFeature(aligned_vis)
detail_vis = DIDF_Encoder.detailFeature(aligned_vis)
```

但你当前 `Restormer_Encoder.forward()` 里已经直接返回了 base/detail，因此需要小改 encoder，让它支持返回 shared feature 后再外部分解

## 3.3 建议修改 `Restormer_Encoder`
添加一个方法：
```python
class Restormer_Encoder(nn.Module):
    ...

    def encode_shared(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        return out_enc_level1

    def decompose(self, shared_feature):
        base_feature = self.baseFeature(shared_feature)
        detail_feature = self.detailFeature(shared_feature)
        return base_feature, detail_feature

    def forward(self, inp_img):
        shared_feature = self.encode_shared(inp_img)
        base_feature, detail_feature = self.decompose(shared_feature)
        return base_feature, detail_feature, shared_feature
```

## 3.4 Alignment Module 的实现方案
### 方案 1：轻量 Offset Alignment
如果你不想额外依赖 `torchvision.ops.DeformConv2d`，可以先实现一个轻量 offset gate：
```python
class FeatureAlignment(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.offset_pred = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 2, 3, 1, 1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, feat_ref, feat_src):
        offset = self.offset_pred(torch.cat([feat_ref, feat_src], dim=1))
        feat_src_warp = self.warp(feat_src, offset)
        feat_src_warp = self.refine(feat_src_warp)
        return feat_ref, feat_src_warp

    def warp(self, x, offset):
        b, c, h, w = x.size()
        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device),
            torch.arange(w, device=x.device),
            indexing='ij'
        )
        grid = torch.stack((xx, yy), dim=-1).float()
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)

        grid = grid + offset.permute(0, 2, 3, 1)
        grid_x = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
        grid_y = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)

        return F.grid_sample(x, grid, mode='bilinear',
                             padding_mode='border',
                             align_corners=True)
```

建议以 **visible 为参考，对齐 infrared**，或者以 **base feature 相关性更高的一方为参考**。

```python
shared_vis, shared_ir_aligned = AlignModule(shared_vis, shared_ir)
```

# 4. 改动二：Base 分支使用 Common/Discrepancy Cross-Attention
## 4.1 为什么改 base fusion？
CDDFuse 中 base feature 代表低频、全局、模态共享信息；原论文希望通过相关性损失让红外和可见光 base feature 更相关。 但如果只强调相关性，容易出现一个问题：
> base 分支过度关注共同背景，忽略红外目标与可见光结构之间的互补差异。

ATFusion 指出，普通 cross-attention 容易提取共同信息，而没有充分考虑 source images 的 discrepancy information，因此提出差异信息注入和交替共同信息注入。
https://arxiv.org/abs/2401.11675

## 4.2 Base Fusion 设计
建议替换：
```python
BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8)
```
为：
```python
BaseFuseLayer = BaseCrossFusion(dim=64, num_heads=8)
```

结构如下：
```txt
B_ir, B_vis
   ├── Common Cross-Attention
   ├── Discrepancy Cross-Attention
   └── Adaptive Gate
          ↓
      B_fused
```

## 4.3 Cross Attention 实现
```python
class CrossAttention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, 1, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.q_dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)
        self.kv_dw = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, q_feat, kv_feat):
        b, c, h, w = q_feat.shape

        q = self.q_dw(self.q(q_feat))
        kv = self.kv_dw(self.kv(kv_feat))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        return self.proj(out)
```
这个写法风格与你当前 `Attention` 和 `AttentionBase` 接近，便于迁移。

## 4.4 Common/Discrepancy Fusion 模块
```python
class BaseCrossFusion(nn.Module):
    def __init__(self, dim=64, num_heads=8):
        super().__init__()
        self.norm_ir = LayerNorm(dim, 'WithBias')
        self.norm_vis = LayerNorm(dim, 'WithBias')

        self.cross_ir_to_vis = CrossAttention(dim, num_heads)
        self.cross_vis_to_ir = CrossAttention(dim, num_heads)

        self.common_proj = nn.Conv2d(dim * 2, dim, 1)
        self.diff_proj = nn.Conv2d(dim * 2, dim, 1)

        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        self.ffn = Mlp(dim, ffn_expansion_factor=1.0)

    def forward(self, b_ir, b_vis):
        b_ir_n = self.norm_ir(b_ir)
        b_vis_n = self.norm_vis(b_vis)

        ir_context = self.cross_ir_to_vis(b_ir_n, b_vis_n)
        vis_context = self.cross_vis_to_ir(b_vis_n, b_ir_n)

        common = self.common_proj(torch.cat([ir_context, vis_context], dim=1))

        diff_ir = torch.abs(b_ir - vis_context)
        diff_vis = torch.abs(b_vis - ir_context)
        discrepancy = self.diff_proj(torch.cat([diff_ir, diff_vis], dim=1))

        gate = self.gate(torch.cat([common, discrepancy], dim=1))
        fused = common + gate * discrepancy

        fused = fused + self.ffn(fused)
        return fused
```

这种设计对应：
- `common`：建模两个模态都存在的低频背景结构；
- `discrepancy`：补充红外热目标、可见光结构差异；
- `gate`：自适应控制差异信息注入强度。

# 5. 改动三：Detail 分支加入 Edge-aware Fusion
## 5.1 为什么 detail 分支要特殊处理？
CDDFuse 中 detail 分支由 INN 提取高频局部信息，用于保留纹理、边缘和模态特异信息。 但是在复杂场景中，直接融合 detail feature 可能引入：
- 可见光噪声；
- 红外伪边缘；
- 融合后纹理过强或过弱。
因此建议保留 INN 结构，但在其前面加入 edge-aware gate。

## 5.2 DetailEdgeFusion 实现
```python
class DetailEdgeFusion(nn.Module):
    def __init__(self, dim=64, num_layers=1):
        super().__init__()
        self.ir_weight = nn.Sequential(
            nn.Conv2d(dim * 2 + 2, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.detail_inn = DetailFeatureExtraction(num_layers=num_layers)

    def forward(self, d_ir, d_vis, grad_ir, grad_vis):
        x = torch.cat([d_ir, d_vis, grad_ir, grad_vis], dim=1)
        w_ir = self.ir_weight(x)
        d = w_ir * d_ir + (1.0 - w_ir) * d_vis
        d = self.detail_inn(d)
        return d
```

其中 `grad_ir` 和 `grad_vis` 可以用 Sobel 梯度：
```python
grad_ir = kornia.filters.sobel(data_IR)
grad_vis = kornia.filters.sobel(data_VIS)
```
不过 `kornia.filters.sobel()` 输出通道可能需要检查，如果维度不匹配，可以转成单通道梯度幅值。

# 6. 改动四：训练流程调整
## 6.1 Stage I：重建 + 对齐 + 分解约束
Stage I 仍然保持 CDDFuse 原始思想：
```txt
IR → Encoder → base_ir/detail_ir → Decoder → rec_ir
VIS → Encoder → base_vis/detail_vis → Decoder → rec_vis
```

但加入 feature alignment：
```shell
shared_ir = DIDF_Encoder.encode_shared(data_IR)
shared_vis = DIDF_Encoder.encode_shared(data_VIS)

shared_vis, shared_ir_aligned = AlignModule(shared_vis, shared_ir)

base_ir, detail_ir = DIDF_Encoder.decompose(shared_ir_aligned)
base_vis, detail_vis = DIDF_Encoder.decompose(shared_vis)
```

Stage I 损失建议：
```python
L_stage1 =
  L_rec_ir
+ L_rec_vis
+ λ_decomp * L_decomp
+ λ_align * L_align
```
其中 CDDFuse 原本使用重建损失和相关性分解损失，让 base feature 相关、detail feature 去相关

## 6.2 Alignment Loss
推荐约束 base feature，而不是 shared/detail feature：
```python
L_align = 1.0 - cc(base_ir, base_vis)
```
因为 base feature 本来就是共享低频结构，对齐它更符合 CDDFuse 假设。
也可以加入梯度结构一致性：
```python
L_align_grad = L1Loss(
    kornia.filters.sobel(base_ir.mean(dim=1, keepdim=True)),
    kornia.filters.sobel(base_vis.mean(dim=1, keepdim=True))
)
```

## 6.3 Stage II：融合训练
Stage II 使用：
```python
shared_ir = DIDF_Encoder.encode_shared(data_IR)
shared_vis = DIDF_Encoder.encode_shared(data_VIS)

shared_vis, shared_ir_aligned = AlignModule(shared_vis, shared_ir)

base_ir, detail_ir = DIDF_Encoder.decompose(shared_ir_aligned)
base_vis, detail_vis = DIDF_Encoder.decompose(shared_vis)

base_fused = BaseFuseLayer(base_ir, base_vis)

grad_ir = get_gradient(data_IR)
grad_vis = get_gradient(data_VIS)
detail_fused = DetailFuseLayer(detail_ir, detail_vis, grad_ir, grad_vis)

fused_img, _ = DIDF_Decoder(None, base_fused, detail_fused)
```

Stage II 损失：
```python
L_stage2 =
  L_intensity
+ λ_grad * L_gradient
+ λ_decomp * L_decomp
+ λ_align * L_align
+ λ_edge * L_edge
```

# 7. 推荐损失函数设计
## 7.1 强度损失
保持 CDDFuse 原始思想：
```python
target_intensity = torch.max(data_IR, data_VIS)
loss_intensity = L1Loss(fused_img, target_intensity)
```
CDDFuse Stage II 中也使用类似最大强度约束来保留显著信息。

## 7.2 梯度损失
```python
grad_fused = get_gradient(fused_img)
grad_target = torch.max(get_gradient(data_IR), get_gradient(data_VIS))
loss_grad = L1Loss(grad_fused, grad_target)
```
CDDFuse Stage II 原本也使用 Sobel 梯度损失来保留边缘和纹理。

## 7.3 分解损失
继续使用原始 CDDFuse 的相关性分解损失：
```python
loss_decomp = (cc(detail_ir, detail_vis) ** 2) / (cc(base_ir, base_vis) + 1.01)
```
原论文中该损失用于让 base features 更相关，detail features 更去相关

## 7.4 差异信息保护损失
为了避免 cross-attention 抹掉红外目标，可以加：
```python
loss_ir_saliency = L1Loss(
    fused_img * ir_mask,
    data_IR * ir_mask
)
```

如果暂时没有语义 mask，可以用红外亮度阈值近似：

```python
ir_mask = (data_IR > data_IR.mean(dim=[2, 3], keepdim=True)).float()
```

# 参考文献
- **STFNet: Self-supervised Transformer for Infrared and Visible Image Fusion**  
    参考点：deformable convolution feature alignment、detail self-attention、saliency cross-attention、频域一致性损失。 [[github.com]](https://github.com/QiaoLiuHit/STFNet)
    
- **ATFusion: An Alternate Cross-Attention Transformer Network for Infrared and Visible Image Fusion**  
    参考点：discrepancy information injection、alternate common information injection、cross-attention 改造。 [[arxiv.org]](https://arxiv.org/abs/2401.11675)
    
- **Infrared and Visible Image Fusion: From Data Compatibility to Task Adaption**  
    参考点：IVIF 领域综述，重点关注数据兼容性、未配准问题、任务适应性和评价体系。