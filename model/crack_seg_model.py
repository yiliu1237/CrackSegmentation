import torch
import torch.nn as nn
import torch.nn.functional as F

# from .backbone import ViTAdapter
from .decoder import SimpleDecoder, SimpleFPN, SimpleDecoderWithLayerNormGelu
from .dinov2.hub.backbones import dinov2_vits14

#
# class CrackModelViTAdapter(nn.Module):
#     def __init__(self, pretrained_backbone_checkpoint_path=None):
#         super().__init__()
#         self.backbone = ViTAdapter(
#             pretrain_size=592,
#             img_size=592,
#             patch_size=16,
#             embed_dim=384,
#             depth=12,
#             num_heads=6,
#             mlp_ratio=4,
#             drop_path_rate=0.2,
#             conv_inplane=64,
#             n_points=4,
#             deform_num_heads=6,
#             cffn_ratio=0.25,
#             deform_ratio=1.0,
#             interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
#             window_attn=[True, True, False, True, True, False,
#                          True, True, False, True, True, False],
#             window_size=[14, 14, None, 14, 14, None,
#                          14, 14, None, 14, 14, None],
#             pretrained=pretrained_backbone_checkpoint_path
#         )
#         self.decoder = SimpleDecoder()
#
#     def forward(self, x):
#         f1, f2, f3, f4 = self.backbone.forward(x)
#         mask = self.decoder.forward(f1, f2, f3, f4, x)
#         return mask



class CrackModelDinoV2(nn.Module):
    def __init__(self, pretrained_backbone_checkpoint_path):
        super().__init__()
        self.backbone = dinov2_vits14(pretrained=False)
        if pretrained_backbone_checkpoint_path:
            state_dict = torch.load(pretrained_backbone_checkpoint_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=True)
        else:
            print('Not Load NINOv2 Checkpoint!')

        self.fpn = SimpleFPN(backbone_channel=384)
        self.decoder = SimpleDecoderWithLayerNormGelu()

    def forward(self, x):
        with torch.no_grad():
            f = self.backbone.get_intermediate_layers(x, n=1, reshape=True)
        # print(f[0].shape)
        f1, f2, f3, f4 = self.fpn.forward(f[0])
        mask = self.decoder.forward(f1, f2, f3, f4, x)
        return mask

    def train(self: nn.Module, mode: bool = True) -> nn.Module:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            if (
                    module is self.backbone
            ):
                module.train(False)
            else:
                module.train(mode)
        return self