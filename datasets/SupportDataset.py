from __future__ import print_function, division
import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class SupportDataset(Dataset):
    def __init__(self, dataset_images_info, root_dir, transform=None):
        self.images_info = []
        self.root_dir = root_dir
        self.images_info = dataset_images_info

        self.transform = transform

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_info = self.images_info[idx]
        img = Image.open(image_info['path']).convert("RGB")

        x, y, w, h = image_info['box']
        cropped_image = img.crop((x, y, x + w, y + h))

        """plt.imshow(cropped_image)
        plt.title(image_info['category_name'])
        plt.show()"""

        if self.transform is not None:
            cropped_image = self.transform(cropped_image)

        y_label = image_info['category_id']

        return cropped_image, y_label


