
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 基础卷积块
# -----------------------------
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super().__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = []
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# -----------------------------
# MixStructureBlock
# -----------------------------
class MixStructureBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        # 大尺度卷积
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, 7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, 5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, 3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, 1),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x



# -----------------------------
# Bottleneck
# -----------------------------
# -----------------------------
# RFGM 模块 (必须放在 BottleNect 之前)
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


# class BottleNect(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         # 1. 局部特征提取 (Patch-Aware 模拟)
#         # 用不同步长的卷积来模拟 2x2 和 4x4 的 Patch 采样
#         self.pa2 = nn.Sequential(
#             nn.AvgPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(dim, dim, 1),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         )
#         self.pa4 = nn.Sequential(
#             nn.AvgPool2d(kernel_size=4, stride=4),
#             nn.Conv2d(dim, dim, 1),
#             nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
#         )
#
#         # 2. 权重生成 Wi (公式 11)
#         self.attn_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
#             nn.ReLU(),
#             nn.Conv2d(dim, dim, 1),
#             nn.Sigmoid()
#         )
#
#         # 3. 原始大核卷积分支
#         ker = 63
#         pad = ker // 2
#         self.dw_conv = nn.Conv2d(dim, dim, ker, padding=pad, groups=dim)
#
#         # 4. GFFN 门控前馈网络 (公式 10/12)
#         self.gffn = nn.Sequential(
#             nn.Conv2d(dim, dim * 2, 1),
#             nn.GELU(),
#             nn.Conv2d(dim * 2, dim, 1)  # 这里简化处理，你可以根据需要加 SimpleGate
#         )
#
#     def forward(self, x):
#         identity = x
#
#         # --- 阶段 1: 获取低频引导 Wi ---
#         # 模拟公式 11: Wi = A(ReLU(DConv(Fl)) + PA4 + PA2)
#         low_feat = x
#         wi = self.attn_conv(self.pa2(low_feat) + self.pa4(low_feat))
#
#         # --- 阶段 2: 引导高频增强 ---
#         # 模拟公式 12: Wi ⊙ Fh
#         high_feat = self.dw_conv(x)
#         modulated_feat = high_feat * wi
#
#         # --- 阶段 3: 最后精修 ---
#         out = self.gffn(modulated_feat)
#
#         return out + identity
# class BottleNect(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         # 1. 局部特征提取 (Patch-Aware 模拟)
#         # 用不同步长的卷积来模拟 2x2 和 4x4 的 Patch 采样
#         self.pa2 = nn.Sequential(
#             nn.AvgPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(dim, dim, 1),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         )
#         self.pa4 = nn.Sequential(
#             nn.AvgPool2d(kernel_size=4, stride=4),
#             nn.Conv2d(dim, dim, 1),
#             nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
#         )
#
#         # 2. 权重生成 Wi (公式 11)
#         self.attn_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
#             nn.ReLU(),
#             nn.Conv2d(dim, dim, 1),
#             nn.Sigmoid()
#         )
#
#         # 3. 原始大核卷积分支
#         ker = 63
#         pad = ker // 2
#         self.dw_conv = nn.Conv2d(dim, dim, ker, padding=pad, groups=dim)
#
#         # 4. GFFN 门控前馈网络 (公式 10/12)
#         self.gffn = nn.Sequential(
#             nn.Conv2d(dim, dim * 2, 1),
#             nn.GELU(),
#             nn.Conv2d(dim * 2, dim, 1)  # 这里简化处理，你可以根据需要加 SimpleGate
#         )
#
#     # def forward(self, x):
#     #     identity = x
#     #
#     #     # --- 阶段 1: 获取低频引导 Wi ---
#     #     # 模拟公式 11: Wi = A(ReLU(DConv(Fl)) + PA4 + PA2)
#     #     low_feat = x
#     #     wi = self.attn_conv(self.pa2(low_feat) + self.pa4(low_feat))
#     #
#     #     # --- 阶段 2: 引导高频增强 ---
#     #     # 模拟公式 12: Wi ⊙ Fh
#     #     high_feat = self.dw_conv(x)
#     #     modulated_feat = high_feat * wi
#     #
#     #     # --- 阶段 3: 最后精修 ---
#     #     out = self.gffn(modulated_feat)
#     #
#     #     return out + identity
#     def forward(self, x):
#         identity = x
#
#         # --- 阶段 1: 获取低频引导 Wi ---
#         # 模拟公式 11: Wi = A(ReLU(DConv(Fl)) + PA4 + PA2)
#         low_feat = x
#         wi = self.attn_conv(self.pa2(low_feat) + self.pa4(low_feat))
#
#         # --- 阶段 2: 引导高频增强 ---
#         # 模拟公式 12: Wi ⊙ Fh
#         high_feat = self.dw_conv(x)
#         modulated_feat = high_feat * wi
#
#         # --- 阶段 3: 最后精修 ---
#         out = self.gffn(modulated_feat)
#
#         return out + identity



class BottleNect(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1. 局部特征提取 (Patch-Aware 模拟)
        # 用不同步长的卷积来模拟 2x2 和 4x4 的 Patch 采样
        self.pa2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, dim, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.pa4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(dim, dim, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

        # 2. 权重生成 Wi (公式 11)
        self.attn_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # 3. 原始大核卷积分支
        ker = 63
        pad = ker // 2
        self.dw_conv = nn.Conv2d(dim, dim, ker, padding=pad, groups=dim)

        # 4. GFFN 门控前馈网络 (公式 10/12)
        self.gffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)  # 这里简化处理，你可以根据需要加 SimpleGate
        )

    # def forward(self, x):
    #     identity = x
    #
    #     # --- 阶段 1: 获取低频引导 Wi ---
    #     # 模拟公式 11: Wi = A(ReLU(DConv(Fl)) + PA4 + PA2)
    #     low_feat = x
    #     wi = self.attn_conv(self.pa2(low_feat) + self.pa4(low_feat))
    #
    #     # --- 阶段 2: 引导高频增强 ---
    #     # 模拟公式 12: Wi ⊙ Fh
    #     high_feat = self.dw_conv(x)
    #     modulated_feat = high_feat * wi
    #
    #     # --- 阶段 3: 最后精修 ---
    #     out = self.gffn(modulated_feat)
    #
    #     return out + identity
    # def forward(self, x):
    #     identity = x
    #
    #     # --- 阶段 1: 获取低频引导 Wi ---
    #     # 模拟公式 11: Wi = A(ReLU(DConv(Fl)) + PA4 + PA2)
    #     low_feat = x
    #     wi = self.attn_conv(self.pa2(low_feat) + self.pa4(low_feat))
    #
    #     # --- 阶段 2: 引导高频增强 ---
    #     # 模拟公式 12: Wi ⊙ Fh
    #     high_feat = self.dw_conv(x)
    #     modulated_feat = high_feat * wi
    #
    #     # --- 阶段 3: 最后精修 ---
    #     out = self.gffn(modulated_feat)
    #
    #     return out + identity

    # def forward(self, x):
    #     identity = x
    #
    #     # --- 阶段 1: 获取低频引导 Wi ---
    #     # 模拟公式 11: Wi = A(ReLU(DConv(Fl)) + PA4 + PA2)
    #     low_feat = x
    #     wi = self.attn_conv(self.pa2(low_feat) + self.pa4(low_feat))
    #
    #     # --- 阶段 2: 引导高频增强 ---
    #     # 模拟公式 12: Wi ⊙ Fh
    #     high_feat = self.dw_conv(x)
    #     modulated_feat = high_feat * wi
    #
    #     # --- 阶段 3: 最后精修 ---
    #     out = self.gffn(modulated_feat)
    #
    #     return out + identity

    def forward(self, x):
        identity = x

        # --- 阶段 1: 获取低频引导 Wi ---
        # 记录原始输入的 H 和 W
        _, _, H, W = x.size()

        low_feat = x
        feat_pa2 = self.pa2(low_feat)
        feat_pa4 = self.pa4(low_feat)

        # 【核心修复】检查 pa2 和 pa4 是否因为整除问题导致尺寸不一致
        if feat_pa4.shape != feat_pa2.shape:
            # 强制将 pa4 对齐到 pa2 的尺寸 (例如 152 补回 154)
            feat_pa4 = F.interpolate(feat_pa4, size=(feat_pa2.shape[2], feat_pa2.shape[3]),
                                     mode='bilinear', align_corners=False)

        # 此时加法绝对安全
        wi = self.attn_conv(feat_pa2 + feat_pa4)

        # --- 阶段 2: 引导高频增强 ---
        high_feat = self.dw_conv(x)

        # 再次确保权重图 wi 与卷积后的特征图尺寸一致
        if wi.shape != high_feat.shape:
            wi = F.interpolate(wi, size=(high_feat.shape[2], high_feat.shape[3]),
                               mode='bilinear', align_corners=False)

        modulated_feat = high_feat * wi

        # --- 阶段 3: 最后精修 ---
        out = self.gffn(modulated_feat)

        return out + identity

# -----------------------------
# BasicLayer
# -----------------------------
class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.blocks = nn.ModuleList([MixStructureBlock(dim) for _ in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# -----------------------------
# PatchEmbed & PatchUnEmbed
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        return self.proj(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = 1
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        return self.proj(x)


# -----------------------------
# SKFusion
# -----------------------------
class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super().__init__()
        self.height = height
        d = max(int(dim / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1).view(B, self.height, C, H, W)
        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))
        out = torch.sum(in_feats * attn, dim=1)
        return out


# -----------------------------
# MixDehazeNet
# -----------------------------
class MixDehazeNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=4,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[1, 1, 2, 1, 1]):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])
        self.patch_merge1 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)
        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])
        self.patch_merge2 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)
        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])
        self.bottleneck = BottleNect(embed_dims[2])
        self.patch_split1 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])
        self.fusion1 = SKFusion(embed_dims[3])
        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])
        self.patch_split2 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        self.fusion2 = SKFusion(embed_dims[4])
        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])
        self.patch_unembed = PatchUnEmbed(patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (4 - h % 4) % 4
        mod_pad_w = (4 - w % 4) % 4
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.bottleneck(x)
        x = self.patch_split1(x)
        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)
        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x


# -----------------------------
# 模型快速生成函数
# -----------------------------
def MixDehazeNet_t():
    return MixDehazeNet(embed_dims=[24, 48, 96, 48, 24], depths=[1, 1, 2, 1, 1])

def MixDehazeNet_s():
    return MixDehazeNet(embed_dims=[24, 48, 96, 48, 24], depths=[2, 2, 4, 2, 2])

def MixDehazeNet_b():
    return MixDehazeNet(embed_dims=[24, 48, 96, 48, 24], depths=[4, 4, 8, 4, 4])

def MixDehazeNet_l():
    return MixDehazeNet(embed_dims=[24, 48, 96, 48, 24], depths=[8, 8, 16, 8, 8])

