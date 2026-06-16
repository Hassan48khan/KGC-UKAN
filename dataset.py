"""Dataset for binary medical image segmentation.

Expected layout (one folder per dataset under --data_dir):

    inputs/
      busi/
        images/  img1.png img2.png ...
        masks/   img1.png img2.png ...   # same basenames as images

Masks are single-channel; foreground > 127 is treated as class 1. All images
and masks are resized to (input_h, input_w). Augmentation (train only):
random horizontal flip, vertical flip, and rotation.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MedicalSegDataset(Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext,
                 input_w, input_h, num_classes=1, augment=False):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.input_w = input_w
        self.input_h = input_h
        self.num_classes = num_classes
        self.augment = augment

    def __len__(self):
        return len(self.img_ids)

    def _aug(self, img, mask):
        if np.random.rand() < 0.5:                       # horizontal flip
            img = img[:, ::-1, :].copy(); mask = mask[:, ::-1].copy()
        if np.random.rand() < 0.5:                       # vertical flip
            img = img[::-1, :, :].copy(); mask = mask[::-1, :].copy()
        if np.random.rand() < 0.5:                       # rotation (90/180/270)
            k = np.random.randint(1, 4)
            img = np.rot90(img, k).copy(); mask = np.rot90(mask, k).copy()
        return img, mask

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        if img is None:
            raise FileNotFoundError(os.path.join(self.img_dir, img_id + self.img_ext))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext),
                          cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(os.path.join(self.mask_dir, img_id + self.mask_ext))

        img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.input_w, self.input_h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self._aug(img, mask)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))           # C,H,W
        mask = (mask > 127).astype(np.float32)[None]             # 1,H,W
        mask = torch.from_numpy(mask)
        return img, mask, {'img_id': img_id}


def list_image_ids(img_dir, img_ext):
    return sorted([os.path.splitext(f)[0] for f in os.listdir(img_dir)
                   if f.endswith(img_ext)])
