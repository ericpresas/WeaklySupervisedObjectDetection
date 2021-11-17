from __future__ import print_function, division
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np


class RegionsDataset(Dataset):
    def __init__(self, dataset_images_info, categories_ids, root_dir, transform=None):
        self.images_info = dataset_images_info
        self.root_dir = root_dir
        self.categories = categories_ids

        self.num_classes = len(self.categories)

        self.transform = transform

        random.shuffle(self.images_info)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_info = self.images_info[idx]
        img = Image.open(image_info['path']).convert("RGB")

        #plt.imshow(img)
        #plt.show()
        cropped_images = []

        for i, annotation in enumerate(image_info['edge_boxes'][:500]):

            x, y, w, h = annotation
            #cropped_image = img[y:y+h, x:x+w]
            cropped_image = img.crop((x, y, x + w, y + h))
            #plt.imshow(cropped_image)
            #plt.show()

            if self.transform is not None:
                cropped_image = self.transform(cropped_image)

            cropped_images.append(cropped_image)

        cropped_images = torch.stack(cropped_images)
        #end = time.time()
        #print(f"Time: {end - start}")

        #y_label = torch.zeros(self.num_classes, dtype=torch.float).scatter_(0, torch.tensor(image_info['category_id'] - 1), value=1)

        return cropped_images, image_info


