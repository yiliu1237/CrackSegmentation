"""
Dataset for Joint Training (Base Model + Refinement Network)

This dataset provides:
    1. Images
    2. Ground truth masks (used as target for BOTH stages)
    3. Optional: Synthetically thickened masks (for training refinement network)

Key Insight:
    - Base model target: GT mask (trained with higher crack_weight for recall)
    - Refinement target: Same GT mask (trained with lower crack_weight for precision)
    - The different loss weights make them learn different behaviors!
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ImgToTensor(object):
    def __call__(self, img):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        return tf(img)


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).long()


class JointCrackDataset(Dataset):
    """
    Dataset for joint training of base segmentation + refinement network.

    Returns:
        image: Input image
        mask: Ground truth mask (target for both base and refinement)
        thick_mask: (Optional) Synthetically thickened version for training refinement
    """

    def __init__(
        self,
        img_dir,
        img_fnames,
        mask_dir,
        mask_fnames,
        image_size,
        training=False,
        raw_size=False,
        repeat_ratio=5,
        use_synthetic_thick=True,
        thick_dilation_iterations=3,
        aug_config=None,
        norm_config=None,
        loss_config=None
    ):
        """
        Args:
            img_dir: Directory containing images
            img_fnames: List of image filenames
            mask_dir: Directory containing masks
            mask_fnames: List of mask filenames
            image_size: Target image size (int or tuple)
            training: Whether in training mode
            raw_size: If True, don't resize images
            repeat_ratio: Number of times to repeat dataset per epoch
            use_synthetic_thick: If True, create synthetic thick masks via dilation
            thick_dilation_iterations: Number of dilation iterations for synthetic thick masks
            aug_config: Augmentation configuration
            norm_config: Normalization configuration
            loss_config: Loss configuration
        """
        self.img_dir = img_dir
        self.img_fnames = sorted(img_fnames)
        self.repeat_ratio = repeat_ratio

        self.mask_dir = mask_dir
        self.mask_fnames = sorted(mask_fnames)

        print("self.img_fnames: ", len(self.img_fnames))
        print("self.mask_fnames: ", len(self.mask_fnames))
        assert len(self.img_fnames) == len(self.mask_fnames)

        self.real_length = len(self.img_fnames)

        self.image_size = image_size
        self.training = training
        self.raw_size = raw_size
        self.use_synthetic_thick = use_synthetic_thick
        self.thick_dilation_iterations = thick_dilation_iterations

        # Augmentation setup
        if aug_config is not None and hasattr(aug_config, 'enabled') and aug_config.enabled:
            aug_list = []
            if hasattr(aug_config, 'random_scale'):
                aug_list.append(albumentations.RandomScale(
                    scale_limit=aug_config.random_scale.range,
                    p=aug_config.random_scale.prob
                ))
            if hasattr(aug_config, 'random_crop'):
                aug_list.append(albumentations.RandomResizedCrop(
                    height=self.image_size,
                    width=self.image_size,
                    scale=tuple(aug_config.random_crop.scale)
                ))
            if hasattr(aug_config, 'motion_blur'):
                aug_list.append(albumentations.MotionBlur(p=aug_config.motion_blur.prob))
            if hasattr(aug_config, 'gaussian_blur'):
                aug_list.append(albumentations.GaussianBlur(p=aug_config.gaussian_blur.prob))
            if hasattr(aug_config, 'color_jitter'):
                aug_list.append(albumentations.ColorJitter(
                    brightness=aug_config.color_jitter.brightness,
                    contrast=aug_config.color_jitter.contrast,
                    saturation=aug_config.color_jitter.saturation,
                    hue=aug_config.color_jitter.hue
                ))
            if hasattr(aug_config, 'rotation'):
                aug_list.append(albumentations.SafeRotate(limit=aug_config.rotation.limit))
            if hasattr(aug_config, 'flips'):
                if aug_config.flips.horizontal:
                    aug_list.append(albumentations.HorizontalFlip())
                if aug_config.flips.vertical:
                    aug_list.append(albumentations.VerticalFlip())
            self.aug = albumentations.Compose(aug_list)
        else:
            # Default augmentation
            self.aug = albumentations.Compose([
                albumentations.RandomScale((-0.7, 0.5), p=0.7),
                albumentations.RandomResizedCrop(
                    height=self.image_size,
                    width=self.image_size,
                    scale=(0.6, 1.0)
                ),
                albumentations.MotionBlur(p=0.2),
                albumentations.GaussianBlur(p=0.2),
                albumentations.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.3,
                    hue=0.3
                ),
                albumentations.SafeRotate(limit=(-90, 90)),
                albumentations.HorizontalFlip(),
                albumentations.VerticalFlip(),
            ])

        self.norm_config = norm_config
        self.loss_config = loss_config

        self.img_totensor = ImgToTensor()
        self.mask_totensor = MaskToTensor()

    def _create_synthetic_thick_mask(self, mask):
        """
        Create a synthetically thickened version of the mask via morphological dilation.

        This simulates what the base model might predict (thick detections).

        Args:
            mask: Original mask (H, W), np.uint8, values 0-255

        Returns:
            thick_mask: Dilated mask (H, W), np.uint8, values 0-255
        """
        # Binarize
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Dilate to make thick
        kernel = np.ones((3, 3), np.uint8)
        thick_mask = cv2.dilate(binary, kernel, iterations=self.thick_dilation_iterations)

        return thick_mask

    def __getitem__(self, index):
        index %= self.real_length

        # Read image
        fname = self.img_fnames[index]
        fpath = os.path.join(self.img_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            raise RuntimeError(f"Failed to load image: {fpath}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read mask
        mname = self.mask_fnames[index]
        mpath = os.path.join(self.mask_dir, mname)
        mask = cv2.imread(mpath, cv2.COLOR_BGR2GRAY)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mpath}")

        # Get threshold value
        binary_threshold = 127
        if self.loss_config is not None and hasattr(self.loss_config, 'binary_threshold'):
            binary_threshold = self.loss_config.binary_threshold

        if self.training:
            # Create synthetic thick mask before augmentation
            if self.use_synthetic_thick:
                thick_mask = self._create_synthetic_thick_mask(mask)
                # Augment both masks together
                transformed = self.aug(image=img, masks=[mask, thick_mask])
                img = transformed['image']
                mask = transformed['masks'][0]
                thick_mask = transformed['masks'][1]
            else:
                transformed = self.aug(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']
                thick_mask = None

            # Convert to tensor
            img = self.img_totensor(img)

            # Binarize mask
            _, mask = cv2.threshold(mask, binary_threshold, 1, cv2.THRESH_BINARY)
            mask = self.mask_totensor(mask).unsqueeze(0)

            if self.use_synthetic_thick and thick_mask is not None:
                _, thick_mask = cv2.threshold(thick_mask, binary_threshold, 1, cv2.THRESH_BINARY)
                thick_mask = self.mask_totensor(thick_mask).unsqueeze(0)
                return img, mask, thick_mask
            else:
                return img, mask, mask  # Use same mask for both if no synthetic thick

        else:
            # Validation/test mode
            if not self.raw_size:
                size = (self.image_size, self.image_size) if isinstance(self.image_size, int) else self.image_size
                img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_CUBIC)

            img = self.img_totensor(img)

            _, mask = cv2.threshold(mask, binary_threshold, 1, cv2.THRESH_BINARY)
            mask = self.mask_totensor(mask).unsqueeze(0)

            return img, mask, mask  # No synthetic thick in validation

    def __len__(self):
        return self.real_length * self.repeat_ratio


if __name__ == "__main__":
    from pathlib import Path

    print("Testing JointCrackDataset...\n")

    # Test dataset
    test_dataset = JointCrackDataset(
        img_dir='./datasets/crack_v1_v2_refined_512_512/train/images',
        img_fnames=[path.name for path in Path('./datasets/crack_v1_v2_refined_512_512/train/images').glob('*.jpg')],
        mask_dir='./datasets/crack_v1_v2_refined_512_512/train/masks',
        mask_fnames=[path.name for path in Path('./datasets/crack_v1_v2_refined_512_512/train/masks').glob('*.png')],
        image_size=512,
        training=True,
        use_synthetic_thick=True,
        thick_dilation_iterations=3
    )

    print(f"Dataset length: {len(test_dataset)}")
    print(f"Real length: {test_dataset.real_length}")

    # Test loading
    img, mask, thick_mask = test_dataset[0]

    print(f"\nSample data shapes:")
    print(f"  Image: {img.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Thick mask: {thick_mask.shape}")

    print(f"\nValue ranges:")
    print(f"  Image: [{img.min():.3f}, {img.max():.3f}]")
    print(f"  Mask: [{mask.min()}, {mask.max()}]")
    print(f"  Thick mask: [{thick_mask.min()}, {thick_mask.max()}]")

    # Check that thick mask is actually thicker
    mask_area = mask.sum().item()
    thick_area = thick_mask.sum().item()
    print(f"\nMask areas:")
    print(f"  Original mask: {mask_area} pixels")
    print(f"  Thick mask: {thick_area} pixels")
    print(f"  Thickness ratio: {thick_area / (mask_area + 1e-6):.2f}x")

    print("\n✓ All tests passed!")
