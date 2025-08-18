# -------------------------------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SEGMENTATION_DATASET(Dataset):
    def __init__(self, image_dir, gt_dir, transform=None):
        self.images = sorted(os.listdir(image_dir))
        self.gt_images = sorted(os.listdir(gt_dir))
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.transform = transform

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])

        try:
            image = Image.open(image_path).convert("RGB")
            gt_image = Image.open(gt_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image pair: {image_path}, {gt_path} - {e}")
            return None  # Optionally, raise instead

        if self.transform:
            image = self.transform(image)
            gt_image = self.transform(gt_image)

        return image, gt_image