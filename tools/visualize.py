from __future__ import division
import copy
import os
from pathlib import Path
from torch import multiprocessing as mp
import cv2
from matplotlib import pyplot as plt
import numpy as np  # 添加numpy导入

import torch
from torch.utils.tensorboard import SummaryWriter

from dataloaders import MaskToTensor, ImgToTensor
from model import CrackModelDinoV3
import utils

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


class TrainConfig:
    #IMAGES_DIR = 'datasets/crack_v2_CSB_original/val/images/'
    #IMAGES_DIR = 'datasets/crack_v1_original/val/images/'
    #IMAGES_DIR = 'datasets/PE_damage_original/test/images/'
    IMAGES_DIR = 'datasets/honeycomb_data2/test/images/'
    SAVE_RESULT = './test_result'  # path to save results
    RESUME_FROM = './checkpoints_stage_1/epoch-20.pth'

    #RESUME_FROM = './checkpoints/combined_512_50epochs_32batches/checkpoints_stage_1/epoch-50.pth'
    OVERLAY = True  # 添加OVERLAY选项，控制是否在原图上叠加裂纹检测结果
    SAVE_MASK_01 = True


class Trainer(object):
    def __init__(self):
        self.config = TrainConfig()
        self.init_writer()
        self.init_model()
        self.test()

    def init_model(self):
        self.log('Initializing model')
        self.model = CrackModelDinoV3(
            dinov3_arch='convnext_base',
            dinov3_ckpt='./pretrained_checkpoint/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth'
        ).cuda()

        self.log(f'Restoring from checkpoint: {self.config.RESUME_FROM}')
        self.log(self.model.load_state_dict(torch.load(self.config.RESUME_FROM, map_location='cpu'), strict=True))

    def init_writer(self):
        self.log('Initializing writer')
        self.writer = SummaryWriter(self.config.SAVE_RESULT)

    def test(self):
        self.model.eval()

        img_totensor = ImgToTensor()
        img_fnames = [path.name for path in Path(self.config.IMAGES_DIR).glob('*.JPG')]
        img_fnames.extend([path.name for path in Path(self.config.IMAGES_DIR).glob('*.jpg')])
        img_fnames.extend([path.name for path in Path(self.config.IMAGES_DIR).glob('*.jpeg')])
        img_fnames.extend([path.name for path in Path(self.config.IMAGES_DIR).glob('*.png')])
        
        for i, fname in enumerate(sorted(img_fnames)):
            fpath = os.path.join(self.config.IMAGES_DIR, fname)
            print(fpath)
            image = cv2.imread(fpath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            # 保存原始图像尺寸
            original_h, original_w = h, w
            # image = cv2.resize(image, (768, 768), interpolation=cv2.INTER_CUBIC)

            with torch.no_grad():
                image = img_totensor(image).squeeze(0)
                patches, patch_locs = utils.get_img_patches_adjusted(image, 512, stride_ratio=0.66)
                patches = patches.cuda(non_blocking=True)
                patch_list = torch.split(patches, 16)
                model_results = []
                for each in patch_list:
                    model_results.extend(torch.sigmoid(self.model(each)).cpu())
                pred_mask = torch.stack(model_results, 0)
                full_mask = utils.merge_pred_patches_adjusted(image, pred_mask, patch_locs)

                self.save_sample(fpath, full_mask.detach().numpy(), name=str(i + 1),
                                 original_size=(original_w, original_h))

    def save_sample(self, img_path, msk_pred, name='', original_size=None):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        os.makedirs(self.config.SAVE_RESULT, exist_ok=True)

        # 调整预测掩码大小以匹配原始图像
        if original_size is not None:
            msk_pred = cv2.resize(msk_pred, original_size, interpolation=cv2.INTER_NEAREST)
        else:
            msk_pred = msk_pred


        BIN_THRESH = getattr(self.config, "BIN_THRESH", 0.5)
        bin_mask = (msk_pred > BIN_THRESH).astype(np.uint8)


        if self.config.SAVE_MASK_01:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            out01 = os.path.join(self.config.SAVE_RESULT, f"{filename}_mask.png")
            print("out01: ", out01)
            cv2.imwrite(out01, bin_mask * 255)  # scale to 0-255 for visibility

        # 创建叠加图像
        if self.config.OVERLAY:
            # 创建红色半透明覆盖层
            overlay = img.copy()
            # 将预测的裂纹区域设置为红色
            #overlay[msk_pred > 0.5] = [255, 0, 0]  # 红色
            overlay[msk_pred > 0.5] = [0, 0, 255]  # blue
            
            # 设置透明度
            alpha = 0.5
            # 叠加原图和红色覆盖层
            overlayed_img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

            # 保存叠加图像
            cv2.imwrite(self.config.SAVE_RESULT + name + '_overlay.jpg', cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR))
            
        else:

            # 保存原始图像和掩码对比图
            _, axs = plt.subplots(1, 2, figsize=(30, 15))
            plt.tight_layout()
            axs = axs.ravel()

            axs[0].axis('off')
            axs[0].imshow(img / 255.)

            axs[1].axis('off')
            axs[1].imshow(msk_pred * 255, cmap='gray')

            plt.savefig(self.config.SAVE_RESULT + name + '.png')
            plt.close()

    def log(self, msg):
        print(f'[] {msg}')


if __name__ == "__main__":
    Trainer()
