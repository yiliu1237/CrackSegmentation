import torch
from model import CrackModelDinoV2


a = CrackModelDinoV2(pretrained_backbone_checkpoint_path='./pretrained_checkpoint/dinov2_vits14_pretrain.pth')
# a = torch.nn.SyncBatchNorm.convert_sync_batchnorm(a)
print(a(torch.randn((1, 3, 448, 448))).shape)