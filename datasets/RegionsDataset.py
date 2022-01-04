from __future__ import print_function, division
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import warnings
warnings.filterwarnings("ignore")
from utils import utils
import numpy as np
import matplotlib.pyplot as plt


class RegionsDataset(Dataset):
    def __init__(self, dataset_images_info, categories_ids, root_dir, transform=None):
        self.images_info = self.filter_small_boxes(dataset_images_info)
        self.root_dir = root_dir
        self.categories = categories_ids

        self.num_classes = len(self.categories)

        self.transform = transform

        random.shuffle(self.images_info)

    @staticmethod
    def filter_small_boxes(images_info):
        filtered_instances = []
        for image_info in images_info:
            image_info['edge_boxes'] = list(utils.filter_boxes(np.array(image_info['edge_boxes']), 100, 500))
            if len(image_info['edge_boxes']) > 0:
                filtered_instances.append(image_info)

        return filtered_instances

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_info = self.images_info[idx]
        img = Image.open(image_info['path']).convert("RGB")

        #plt.imshow(img)
        #plt.show()
        cropped_images = []

        #image_info['edge_boxes'] = list(utils.filter_boxes(np.array(image_info['edge_boxes']), 100, 500))

        if len(image_info['edge_boxes']) > 500:
            image_info['edge_boxes'] = image_info['edge_boxes'][:500]

        if len(image_info['edge_boxes']) > 0:
            for i, annotation in enumerate(image_info['edge_boxes']):

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


