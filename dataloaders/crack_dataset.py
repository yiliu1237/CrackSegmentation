import os

import albumentations
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ImgToTensor(object):
    def __call__(self, img):
        tf = transforms.Compose([transforms.ToTensor(),                                                                                    
                                 transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        return tf(img)


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).long()


class CrackDataset(Dataset):
    """ dataset class for Crack datasets
    """
    
    def __init__(self, img_dir, img_fnames, mask_dir, mask_fnames, image_size=(448, 448), training=False, raw_size=False, repeat_ratio=5):
        self.img_dir = img_dir
        self.img_fnames = sorted(img_fnames)
        self.repeat_ratio = repeat_ratio

        self.mask_dir = mask_dir
        self.mask_fnames = sorted(mask_fnames)

        # print(self.mask_dir)
        # print(len(self.mask_fnames))
        # print(len(self.img_fnames))
        
        assert len(self.img_fnames) == len(self.mask_fnames)

        self.real_length = len(self.img_fnames)
        # for a, b in zip(self.img_fnames, self.mask_fnames):
        #     assert a[:-3] == b[:-3]

        self.image_size = image_size
        self.training = training
        self.raw_size = raw_size

        self.aug = albumentations.Compose([
                            albumentations.RandomScale((-0.7, 0.5), p=0.7),
                            albumentations.RandomResizedCrop(
                                height=self.image_size[0],
                                width=self.image_size[1],
                                scale=(0.6, 1.0)
                            ),
                            albumentations.MotionBlur(p=0.2),
                            albumentations.GaussianBlur(p=0.2),
                            albumentations.ColorJitter(brightness=0.25,
                                                       contrast=0.25,
                                                       saturation=0.3,
                                                       hue=0.3,),
                            albumentations.SafeRotate(limit=(-90, 90)),
                            albumentations.HorizontalFlip(),
                            albumentations.VerticalFlip(),
                            ])

        self.img_totensor = ImgToTensor()
        self.mask_totensor = MaskToTensor()
                
    def __getitem__(self, index):
        index %= self.real_length
        # read a image given a random integer index
        fname = self.img_fnames[index]
        fpath = os.path.join(self.img_dir, fname)
        #print(fpath)
        img = cv2.imread(fpath) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # H,W,3 np.uint8

        mname = self.mask_fnames[index]
        mpath = os.path.join(self.mask_dir, mname)
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)                                    # H,W, np.uint8

        if self.training:
            # image augmentation
            # # print(img.shape, mask.shape)
            # img  = cv2.resize(img, self.image_size, interpolation=cv2.INTER_CUBIC)     # (256,256,3) np.uint8
            # mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_CUBIC)    # (256,256) np.uint8
            # print(img.shape, mask.shape)
            transformed = self.aug(image=img, mask=mask)
            img  = transformed['image']                                               # (256,256,3) np.uint8
            mask = transformed['mask']                                                # (256,256) np.uint8
            
            # binarize segmentation

            # totensor
            img = self.img_totensor(img)

            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            mask = self.mask_totensor(mask).unsqueeze(0)

            return img, mask
        else:
            if not self.raw_size:
                img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_CUBIC)     # (256,256,3) np.uint8
                mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_CUBIC)    # (256,256) np.uint8
            img = self.img_totensor(img)
                
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            mask = self.mask_totensor(mask).unsqueeze(0)

            return img, mask

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.real_length * self.repeat_ratio


if __name__ == "__main__":
    from pathlib import Path
    test_dataset = CrackDataset(img_dir='./datasets/CRACK500/train/images',
                 img_fnames=[path.name for path in Path('./datasets/CRACK500/train/images').glob('*.jpg')],
                 mask_dir='./datasets/CRACK500/train/masks',
                 mask_fnames=[path.name for path in Path('./datasets/CRACK500/train/masks').glob('*.png')],
                 training=True)
    for i in range(len(test_dataset)):
        print(test_dataset[i][0].shape)
