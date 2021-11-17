from __future__ import print_function, division
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ClassificationDataset(Dataset):
    def __init__(self, dataset_images_info, categories_ids, root_dir, support_images=None, transform=None):
        self.images_info = dataset_images_info
        self.root_dir = root_dir
        self.categories = categories_ids

        self.support_images = support_images
        if self.support_images is not None:
            self.images_info += self.support_images

        self.num_classes = len(self.categories)

        self.transform = transform

        random.shuffle(self.images_info)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_info = self.images_info[idx]
        img = Image.open(image_info['path']).convert("RGB")

        # If is support image loads "true" label.
        if 'box' in image_info:
            x, y, w, h = image_info['box']
            img = img.crop((x, y, x + w, y + h))
            # TODO:
            y_label = image_info['category_id']
        else:
            y_label = torch.from_numpy(image_info['pseudo_label'])

        if self.transform is not None:
            img = self.transform(img)

        return img, y_label


