

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



截至 2026 年 9 月，红外可见光图像融合领域还没有一个严格统一、可信的 Mamba 排行榜。不同论文使用的数据划分、Y/RGB 通道、指标实现和再训练策略差异较大，因此“指标领先”应理解为论文各自协议下表现领先，而不是绝对排名。

结合发表质量、公开代码、测试覆盖范围和技术新颖性，目前值得重点关注的模型如下。

第一梯队：直接生成红外可见光融合图像
模型	时间/发表	主要测试集	核心特点	推荐程度SFMFusion	IEEE TIP 2025	6 个多模态融合数据集	空间频率增强 Mamba、三分支重建、动态融合	很高
FusionMamba	Visual Intelligence，2025 版本	IR-VIS 与多类医学融合集	动态卷积、通道注意力、跨模态 Mamba	很高
MambaDFuse	2024	MSRS、RoadScene、M3FD	CNN+Mamba、浅层通道交换、深层 M3 Block	很高，适合作为基线
PMKFuse	Optics & Laser Technology 2025	红外可见光融合集	并行 Mamba-KAN、轻量异构多分支	较高
MPMamba-D	Remote Sensing 2026	MSRS	多路径 Mamba、双路径注意力融合	较高
WMambaFuse	Scientific Reports 2026	TNO、RoadScene、MSRS	小波域 Mamba、空间频率双分支	较高
CAWM-Mamba	2026，当前为 arXiv	标准融合集、恶劣天气数据	融合与复合天气恢复统一建模	前沿但需谨慎
1. SFMFusion

全称为 Spatial-Frequency Enhanced Mamba for Multi-Modal Image Fusion，已被 IEEE Transactions on Image Processing 接收，是目前最值得关注的 Mamba 图像融合模型之一。论文在 6 个多模态融合数据集上进行实验，并报告优于大部分同期方法。官方仓库已经提供代码、预训练权重和 MSRS 测试样例。

关键设计
三分支结构：同时执行模态重建和图像融合，通过重建任务约束网络保留两幅源图像的完整内容。
SFMB：Spatial-Frequency Enhanced Mamba Block，同时补充 Mamba 的空间感知和频率感知。
DFMB：Dynamic Fusion Mamba Block，动态融合不同模态和不同分支的特征。
对 Mamba 容易损失局部高频纹理的问题进行了专门处理。
评价

如果你准备在 CDDFuse 基础上引入 Mamba，SFMFusion 应当被视为当前优先级最高的对比方法之一。它与 CDDFuse 都重视高低频信息，只是 SFMFusion 使用空间频率增强 Mamba 与重建辅助任务完成特征保留。

2. FusionMamba

FusionMamba 使用改进的 Visual State Space Model，重点解决原始 Mamba 局部纹理建模不足和通道冗余问题。论文覆盖红外可见光、CT-MRI、PET-MRI、SPECT-MRI 和生物医学图像融合，并提供了公开代码及红外可见光预训练权重。

关键设计
DFEM 动态特征增强模块：
动态卷积捕获局部纹理；
通道注意力减少冗余；
Mamba 建立全局依赖。
CMFM 跨模态融合 Mamba：
建模红外与可见光特征的相关性；
强化互补信息；
抑制重复及冲突特征。
DFFM 动态融合模块：组合模态内增强和模态间融合。
评价

FusionMamba 的结构比 MambaDFuse更强调动态卷积和跨模态相关性。如果你关注 MSRS 上的纹理、热目标保留以及多数据集泛化，它是很有价值的复现对象。

3. MambaDFuse

MambaDFuse 是当前 Mamba 融合研究中较成熟、代码较完整、引用较多的代表性基线。它在 MSRS、RoadScene 和 M3FD 上进行红外可见光融合实验，并验证了融合图像的目标检测性能。

关键设计
CNN 提取浅层局部特征；
Mamba 提取高层全局特征；
浅层使用 Channel Exchange；
深层使用 Multi-modal Mamba，也就是 M3 Block；
最后通过逆向特征变换重建融合图像。
评价

它未必是当前指标最高的模型，但具有三个优势：

代码和权重已公开；
模型结构清晰，易于改造；
与 CDDFuse 的双阶段、高低层特征逻辑比较接近。

因此，它更适合作为可靠基线，而不应直接作为现阶段最高性能上限。

4. PMKFuse

PMKFuse 使用并行 Mamba-KAN 异构结构，发表于 2025 年的 Optics and Laser Technology。论文重点不是堆叠更大的 Mamba，而是在轻量化条件下结合不同特征建模机制。

关键设计
PCVM 分支：多通道并行 Cross-Vision Mamba，负责全局和跨模态关系。
PKAGN 分支：基于 KAN 的注意力模块，负责更灵活的局部非线性表达。
使用强度、梯度和特征分解组成的复合损失。
强调较少参数下保持较高的客观指标和视觉质量。
评价

如果你关注：

模型参数量；
推理速度；
有限 GPU 部署；
Mamba 与其他新型网络的混合设计；

PMKFuse 的优先级很高。但其官方仓库规模和社区验证程度目前低于 MambaDFuse、FusionMamba。

5. MPMamba-DMAF

2026 年发表的 Mamba-Based Infrared and Visible Images Fusion Method，主要包含 Multi-Path Mamba 和 Dual-path Mamba Attention Fusion 两部分。论文在 MSRS 上报告 EN、SF、MI 等指标的领先表现，并验证融合结果对目标检测的帮助。

关键设计
MPMamba：从多条路径扫描二维特征，降低单一扫描顺序带来的方向偏置。
DMAF：双路径 Mamba 注意力融合，同时校准红外热目标与可见光纹理。
使用双路径特征解耦和动态注意力平衡两种模态的信息。
评价

该方法比较新，优势主要集中在 MSRS。由于尚缺少与 SFMFusion 相同规模的跨数据集验证，现阶段应视作较新的有竞争力方法，而非已经确立的绝对 SOTA。

6. WMambaFuse

WMambaFuse 于 2026 年发表于 Scientific Reports，测试覆盖 TNO、RoadScene 和 MSRS，论文报告在 AG、VIF、SSIM、SCD 等指标上达到最好或接近最好。

关键设计
Swin Transformer 提取多尺度双模态特征；
Spatial Attention Fusion Module 处理空间区域显著性；
Wavelet-domain Mamba Fusion Module 在小波域融合全局语义和局部细节；
解码器保持不同尺度特征的一致性。
评价

WMambaFuse 的核心优势是空间域和小波频域同时建模，能够减少普通 Mamba 对边缘及纹理的遗忘。对于从 CDDFuse 高低频分解思路继续扩展，它比纯空间 Mamba 更有参考价值。

不过，它同时使用 Swin Transformer、Mamba 和循环解码结构，模型复杂度和实际部署开销需要单独核查。

7. CAWM-Mamba

CAWM-Mamba 于 2026 年 3 月发布，针对普通融合模型在雾、雨、雪等复合退化条件下性能下降的问题，将图像融合与天气恢复放入统一模型。当前主要是前沿预印本，成熟度低于已正式发表的方法。

关键设计
WAPM：感知天气退化并增强可见光特征；
CFIM：完成异构模态对齐与互补特征交换；
WSSB：在小波空间分离不同频率的退化；
Freq-SSM：建模具有方向性的高频天气退化。
评价

如果研究目标是正常对齐的 MSRS，CAWM-Mamba 不一定是首选；如果目标涉及：

夜间雾天；
雨雾组合；
雪雾组合；
恶劣天气下的检测和分割；

它代表了比普通融合指标更前沿的方向。

8. 需要区分的相关模型

以下模型虽然也属于 RGB-IR Mamba，但主要输出检测结果，而不是融合图像，因此不能直接与 CDDFuse、MambaDFuse 的 EN、MI、VIF、QAB/F 等指标进行比较。

WaveMamba

ICCV 2025 的 RGB-红外目标检测模型。它使用 DWT 分解高低频特征，低频通过 Mamba 和门控注意力融合，高频采用绝对最大响应保留。论文报告四个检测基准上平均 mAP 提升约 4.5%。

MSDF-Mamba

针对无人机可见光和红外图像的未对齐问题，增加 Mutual-Spectrum Deformable Alignment 和 Selective Scan Fusion。在 DroneVehicle 和 DVTOD 上分别报告 82.5% 和 86.4% mAP。

这两者特别适合评估：

融合特征是否真正有助于目标检测，而不是只让 EN、SF 等像素统计指标变高。

综合推荐顺序

如果你的目标是继续研究 CDDFuse，并在标准红外可见光融合数据集上比较，我建议优先顺序为：

SFMFusion：当前最值得重点研究的正式发表模型
FusionMamba：动态局部增强和跨模态状态融合
MambaDFuse：代码成熟、结构清晰的基础基线
WMambaFuse：空间与小波频域联合建模
PMKFuse：轻量化 Mamba-KAN 异构设计
MPMamba-DMAF：新型多路径二维扫描
CAWM-Mamba：复杂天气融合与恢复方向
对你当前工作的具体建议

如果你计划在 CDDFuse 上引入 Mamba，较合理的方案不是直接把 Lite Transformer 全部换成 Mamba，而是：

保留 INN Detail Branch；
在 Base Branch 中用二维多方向 Mamba 建模低频全局信息；
在融合阶段加入 Cross-modal Mamba；
增加 DWT/FFT 频率增强分支；
使用局部卷积补偿 Mamba 的像素遗忘；
同时测试 MSRS、RoadScene、M3FD 和 TNO；
统一指标代码后比较 EN、SD、SF、MI、SCD、VIF、QAB/F、SSIM；
补充 YOLO 检测或语义分割结果。

简要结论：目前直接融合任务中，SFMFusion、FusionMamba、MambaDFuse 和 WMambaFuse 最具代表性；如果强调轻量化选择 PMKFuse，如果强调恶劣天气选择 CAWM-Mamba，如果强调下游检测则重点参考 WaveMamba。