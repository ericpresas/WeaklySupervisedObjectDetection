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
    def __init__(self, dataset_images_info, categories_ids, root_dir, transform=None):
        self.images_info = []
        self.root_dir = root_dir
        self.categories = []
        for category, content in dataset_images_info.items():
            self.categories.append(category)
            for image_path in content:
                self.images_info.append({
                    "path": image_path,
                    "category_idx": categories_ids[category]
                })

        self.num_classes = len(self.categories)

        self.transform = transform

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_info = self.images_info[idx]
        img = Image.open(image_info['path']).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        y_label = image_info['category_idx']

        return img, y_label


