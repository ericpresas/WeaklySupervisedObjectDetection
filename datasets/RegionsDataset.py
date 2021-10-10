from __future__ import print_function, division
import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from models import FeatureExtractor
from torchvision import transforms, utils
import json
import random
import matplotlib.pyplot as plt
import itertools
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class RegionsDataset(Dataset):
    def __init__(self, dataset_images_info, categories_ids, root_dir, type='train', level='region', transform=None):
        self.images_info = []
        self.root_dir = root_dir
        self.categories = []
        self.level = level
        for category, content in dataset_images_info[type].items():
            self.categories.append(category)
            if self.level == 'region':
                for object_proposal in content:
                    image_path = object_proposal['path']
                    fake_annotations = object_proposal['fake_annotations']
                    self.images_info.append({
                        "path": image_path,
                        "fake_annotations": fake_annotations,
                        "category_idx": categories_ids[category]
                    })
            else:
                for key, group in itertools.groupby(content, lambda x: x['path']):
                    group = list(group)
                    self.images_info.append({
                        "path": group[0]['path'],
                        "fake_annotations": [element['fake_annotations'] for element in group],
                        "category_idx": categories_ids[category]
                    })

        self.num_classes = len(self.categories)

        self.transform = transform

        random.shuffle(self.images_info)
        self.feature_extractor = FeatureExtractor()

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_info = self.images_info[idx]
        img = Image.open(image_info['path']).convert("RGB")
        cropped_images = []

        for i, annotation in enumerate(image_info['fake_annotations']):
            x, y, w, h = annotation
            cropped_image = img.crop((x, y, x + w, y + h))

            if self.transform is not None:
                cropped_image = self.transform(cropped_image)

            if i == 0:
                cropped_images = cropped_image.unsqueeze_(0)
            else:
                cropped_images = torch.cat((cropped_images, cropped_image.unsqueeze_(0)), 0)

        y_label = torch.zeros(self.num_classes, dtype=torch.float).scatter_(0, torch.tensor(image_info['category_idx']), value=1)

        return cropped_images, y_label


