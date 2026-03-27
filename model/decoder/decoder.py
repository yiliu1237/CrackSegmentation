from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SimpleFPN(nn.Module):
    """Simple Feature Pyramid Network for ViTDet."""

    def __init__(self,backbone_channel: int) -> None:
        super().__init__()
        self.backbone_channel = backbone_channel
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2),
            LN2d(self.backbone_channel // 2),
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2,
                               self.backbone_channel // 4, 2, 2),
            LN2d(self.backbone_channel // 4),
            nn.GELU(),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2),
            LN2d(self.backbone_channel // 2),
            nn.GELU(),
        )
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x) -> List:
        """Forward function.

        Args:
            x (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            List: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        outputs = list()
        outputs.append(self.fpn1(x))
        outputs.append(self.fpn2(x))
        outputs.append(self.fpn3(x))
        outputs.append(self.fpn4(x))
        return outputs


class SimpleDecoderWithLayerNormGelu(nn.Module):
    def __init__(self,
                 backbone_dims=(384, 384, 384 // 2, 384 // 4),
                 decoder_dims=(256, 128, 64, 32, 32)):
        super().__init__()
        self.compress_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone_dims[i], decoder_dims[i], kernel_size=(1, 1), padding=0, bias=False),
                LN2d(decoder_dims[i]),
                nn.GELU(),
                nn.Conv2d(decoder_dims[i], decoder_dims[i], kernel_size=(3, 3), padding=1, bias=False),
                LN2d(decoder_dims[i]),
                nn.GELU()
            ) for i in range(4)
        ])

        self.upsample_2x = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            for _ in range(4)
        ])

        self.merge_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(decoder_dims[i] + decoder_dims[i + 1], decoder_dims[i + 1],
                          kernel_size=(3, 3), padding=1, bias=False),
                LN2d(decoder_dims[i + 1]),
                nn.GELU()
            ) for i in range(4)
        ])

        self.x_conv = nn.Sequential(
            nn.Conv2d(3, decoder_dims[4], kernel_size=(3, 3), stride=2, padding=1, bias=False),
            LN2d(decoder_dims[4]),
            nn.GELU(),
            nn.Conv2d(decoder_dims[4], decoder_dims[4], kernel_size=(3, 3), stride=1, padding=1, bias=False),
            LN2d(decoder_dims[4]),
            nn.GELU(),
            nn.Conv2d(decoder_dims[4], decoder_dims[4], kernel_size=(3, 3), stride=1, padding=1, bias=False),
            LN2d(decoder_dims[4]),
            nn.GELU()
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(decoder_dims[4], decoder_dims[4] // 2, kernel_size=(3, 3), padding=1, bias=False),
            LN2d(decoder_dims[4] // 2),
            nn.GELU()
        )

        self.projection = nn.Conv2d(decoder_dims[4] // 2, 1, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, f_1_4, f_1_8, f_1_16, f_1_32, x):
        f_1_32 = self.compress_conv[0](f_1_32)
        f_1_16 = self.compress_conv[1](f_1_16)
        f_1_8 = self.compress_conv[2](f_1_8)
        f_1_4 = self.compress_conv[3](f_1_4)

        f = self.upsample_2x[0](f_1_32)
        f = self.merge_conv[0](
            torch.cat([f, f_1_16], dim=1)
        )
        f = self.upsample_2x[1](f)
        f = self.merge_conv[1](
            torch.cat([f, f_1_8], dim=1)
        )
        f = self.upsample_2x[2](f)
        f = self.merge_conv[2](
            torch.cat([f, f_1_4], dim=1)
        )

        f = self.upsample_2x[3](f)
        f_1_2 = F.interpolate(self.x_conv(x), size=f.shape[-2:], mode='bicubic', align_corners=False)

        f = self.merge_conv[3](
            torch.cat([f, f + f_1_2], dim=1)
        )
        f = F.interpolate(f, size=x.shape[-2:], mode='bicubic', align_corners=False)
        f = self.output_conv(f)
        return self.projection(f)


class SimpleDecoder(nn.Module):
    def __init__(self,
                 backbone_dims=(384, 384, 384, 384),
                 decoder_dims=(256, 128, 64, 32, 16)):
        super().__init__()
        self.compress_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone_dims[i], decoder_dims[i], kernel_size=(1, 1), padding=0, bias=False),
                nn.BatchNorm2d(decoder_dims[i]),
                nn.ReLU(True),
                nn.Conv2d(decoder_dims[i], decoder_dims[i], kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(decoder_dims[i]),
                nn.ReLU(True)
            ) for i in range(4)
        ])

        self.upsample_2x = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            for _ in range(4)
        ])

        self.merge_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(decoder_dims[i] + decoder_dims[i + 1], decoder_dims[i + 1],
                          kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(decoder_dims[i + 1]),
                nn.ReLU(True)
            ) for i in range(4)
        ])

        self.x_conv = nn.Sequential(
            nn.Conv2d(3, decoder_dims[4], kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dims[4]),
            nn.ReLU(True),
            nn.Conv2d(decoder_dims[4], decoder_dims[4], kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dims[4]),
            nn.ReLU(True)
        )

        self.output_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_dims[4], decoder_dims[4], kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(decoder_dims[4]),
            nn.ReLU(True)
        )

        self.projection = nn.Conv2d(decoder_dims[4], 1, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, f_1_4, f_1_8, f_1_16, f_1_32, x):
        f_1_32 = self.compress_conv[0](f_1_32)
        f_1_16 = self.compress_conv[1](f_1_16)
        f_1_8 = self.compress_conv[2](f_1_8)
        f_1_4 = self.compress_conv[3](f_1_4)
        f_1_2 = self.x_conv(x)

        f = self.upsample_2x[0](f_1_32)
        f = self.merge_conv[0](
            torch.cat([f, f_1_16], dim=1)
        )
        f = self.upsample_2x[1](f)
        f = self.merge_conv[1](
            torch.cat([f, f_1_8], dim=1)
        )
        f = self.upsample_2x[2](f)
        f = self.merge_conv[2](
            torch.cat([f, f_1_4], dim=1)
        )
        f = self.upsample_2x[3](f)
        f = self.merge_conv[3](
            torch.cat([f, f_1_2], dim=1)
        )

        f = self.output_conv(f)
        return self.projection(f)