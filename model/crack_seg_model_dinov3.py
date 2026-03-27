#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

from .yolox.network_blocks import BaseConv, CSPLayer, DWConv
from .dinov3.hub import backbones as dinov3_backbones


class YOLOPAFPNDinoV3(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=None,
        width=1.0,
        dinov3_arch="vit_7b",
        dinov3_ckpt=None,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert width == 1, "width should be 1 for now"

        self.dinov3_arch = dinov3_arch
        self.dinov3_type, self.dinov3_size = dinov3_arch.split("_")
        self.backbone_dinov3 = None
        if self.dinov3_type == 'convnext':
            if self.dinov3_size == 'tiny':
                self.backbone_dinov3 = dinov3_backbones.dinov3_convnext_tiny(pretrained=False)
                if depth is None:
                    depth = 0.33
            elif self.dinov3_size == 'small':
                self.backbone_dinov3 = dinov3_backbones.dinov3_convnext_small(pretrained=False)
                if depth is None:
                    depth = 0.67
            elif self.dinov3_size == 'base':
                self.backbone_dinov3 = dinov3_backbones.dinov3_convnext_base(pretrained=False)
                if depth is None:
                    depth = 1.0
            elif self.dinov3_size == 'large':
                self.backbone_dinov3 = dinov3_backbones.dinov3_convnext_large(pretrained=False)
                if depth is None:
                    depth = 1.33

            # self.backbone_feature_downsample = nn.Identity()
            self.backbone_feature_upsample = nn.Identity()
            self.backbone_dims = self.backbone_dinov3.embed_dims
        elif self.dinov3_type == 'vit':
            if self.dinov3_size == 's':
                self.backbone_dinov3 = dinov3_backbones.dinov3_vits16(pretrained=False)
                if depth is None:
                    # todo 是不是有点小？
                    depth = 0.33
            elif self.dinov3_size == 's+':
                self.backbone_dinov3 = dinov3_backbones.dinov3_vits16plus(pretrained=False)
                if depth is None:
                    # todo 是不是有点小？
                    depth = 0.33
            elif self.dinov3_size == 'b':
                self.backbone_dinov3 = dinov3_backbones.dinov3_vitb16(pretrained=False)
                if depth is None:
                    # todo 是不是有点小？
                    depth = 0.67
            elif self.dinov3_size == 'l':
                self.backbone_dinov3 = dinov3_backbones.dinov3_vitl16(pretrained=False)
                if depth is None:
                    depth = 1.00
            elif self.dinov3_size == 'h+':
                self.backbone_dinov3 = dinov3_backbones.dinov3_vith16plus(pretrained=False)
                if depth is None:
                    depth = 1.33
            elif self.dinov3_size == '7b':
                self.backbone_dinov3 = dinov3_backbones.dinov3_vit7b16(pretrained=False)
                if depth is None:
                    # todo 是不是有点小？
                    depth = 1.67
            # self.backbone_feature_downsample = nn.Upsample(scale_factor=0.5, mode="bilinear")
            self.backbone_feature_upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            self.backbone_dims = [self.backbone_dinov3.embed_dim] * 4
        else:
            raise ValueError('Error dinov3_arch: %s' % dinov3_arch)

        assert dinov3_ckpt is not None
        weights = torch.load(dinov3_ckpt, map_location=torch.device("cpu"), weights_only=True)
        self.backbone_dinov3.load_state_dict(weights, strict=True)

        self.layers_to_use = [m * self.backbone_dinov3.n_blocks // 4 - 1 for m in range(1, 5)]

        assert len(self.backbone_dims) == 4
        in_channels = self.backbone_dims[1:]
        self.in_channels = in_channels

        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.upsample4x = nn.Upsample(scale_factor=4, mode="bilinear")

        if self.dinov3_type == 'convnext':
            self.bu_conv_extra = nn.Identity()
        else:
            self.bu_conv_extra = Conv(
                int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
            )

        self.backbone_dinov3.requires_grad_(False)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        with torch.no_grad():
            # 在当前框架输入就是norm后的0-1的了 不再需要这个
            # input = (input[:, [2, 1, 0], :, :].contiguous() - self.pixel_mean) / self.pixel_std
            x3, x2, x1, x0 = self.backbone_dinov3.get_intermediate_layers(input, n=self.layers_to_use, reshape=True)
        # todo vit最后一层的特征被resize后再用是最好的吗
        # x0 = self.backbone_feature_downsample(x0)  # 512/16
        # print(x2.shape)
        x2 = self.backbone_feature_upsample(x2)
        # print(x2.shape)

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32or16
        if self.dinov3_type == 'convnext':
            f_out0 = self.upsample(fpn_out0)  # 512/16
        else:
            f_out0 = fpn_out0  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        fpn_out0 = self.bu_conv_extra(fpn_out0)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        if self.dinov3_type == 'vit':
            x3 = self.upsample4x(x3)
        outputs = (x3, pan_out2, pan_out1, pan_out0)
        return outputs

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            if module is self.backbone_dinov3:
                module.train(False)
            else:
                module.train(mode)
        return self


class UpperDecoder(nn.Module):
    def __init__(self, input_channels, depthwise=False):
        super().__init__()

        self.input_channels = input_channels

        Conv = DWConv if depthwise else BaseConv

        self.compress_conv = nn.ModuleList([
            Conv(input_channels[i], int(input_channels[i] // 2), 1, 1)
            for i in range(4)
        ])

        self.upsample_2x = nn.Upsample(scale_factor=2, mode='bilinear')

        self.merge_conv = nn.ModuleList([
            Conv(int((input_channels[i] + input_channels[i + 1]) // 2), int(input_channels[i + 1] // 2), 1, 1) for i in range(3)
        ])

        self.output_conv = nn.Sequential(
            Conv(int(input_channels[3] // 2), int(input_channels[3] // 4), 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.projection = nn.Conv2d(int(input_channels[3] // 4), 1, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, f_1_4, f_1_8, f_1_16, f_1_32):
        f_1_32 = self.compress_conv[0](f_1_32)
        f_1_16 = self.compress_conv[1](f_1_16)
        f_1_8 = self.compress_conv[2](f_1_8)
        f_1_4 = self.compress_conv[3](f_1_4)
        # print(f_1_32.shape, f_1_16.shape, f_1_8.shape, f_1_4.shape)

        f = self.upsample_2x(f_1_32)
        f = self.merge_conv[0](
            torch.cat([f, f_1_16], dim=1)
        )
        f = self.upsample_2x(f)
        f = self.merge_conv[1](
            torch.cat([f, f_1_8], dim=1)
        )
        f = self.upsample_2x(f)
        f = self.merge_conv[2](
            torch.cat([f, f_1_4], dim=1)
        )
        f = self.upsample_2x(f)

        f = self.output_conv(f)
        return self.projection(f)


class CrackModelDinoV3(nn.Module):
    def __init__(self, dinov3_arch="vit_7b", dinov3_ckpt=None, depthwise=False):
        super().__init__()
        self.encoder = YOLOPAFPNDinoV3(dinov3_arch=dinov3_arch, dinov3_ckpt=dinov3_ckpt, depthwise=depthwise)
        self.decoder = UpperDecoder(input_channels=self.encoder.backbone_dims[::-1], depthwise=depthwise)

    def forward(self, x):
        f1_4, f1_8, f1_16, f1_32 = self.encoder(x)
        mask = self.decoder.forward(f1_4, f1_8, f1_16, f1_32)
        return mask


if __name__ == "__main__":
    model = CrackModelDinoV3(dinov3_arch='vit_s+', dinov3_ckpt=r'D:\Download\Other\dinov3_pretrained\DINOv3 ViT LVD-1689M\dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth')
    a = model.forward(torch.randn(1, 3, 640, 640))
    print(a.shape)
