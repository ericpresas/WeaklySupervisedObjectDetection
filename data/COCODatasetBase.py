from pycocotools.coco import COCO
import requests
import os
import random
import json
from tqdm import tqdm
from skimage import io
from PIL import Image


class MaxValueDatasetReached(Exception):
    " Raised when max defined value of samples is reached. "
    pass

class MaxImagesDatasetReached(Exception):
    " Raised when no more samples in dataset. "
    pass


class COCODatasetBase(object):
    def __init__(self, data_path: str, data_train_id: int, data_val_id: int, categories: list, max_samples_category: int=50, percentages_dataset: dict=None):

        self.data_path = data_path
        self.data_train_id = data_train_id
        self.data_val_id = data_val_id

        self.ann_file_train = f"{self.data_path}/annotations/instances_{self.data_train_id}.json"
        self.ann_file_val = f"{self.data_path}/annotations/instances_{self.data_val_id}.json"

        if percentages_dataset is None:
            percentages_dataset = {
                "train": 0.7,
                "val": 0.2,
                "test": 0.1
            }

        self.num_images = {
            'train': int(max_samples_category * percentages_dataset['train']),
            'val': int(max_samples_category * percentages_dataset['val'])
        }

        self.num_images['test'] = max_samples_category - (self.num_images['train'] + self.num_images['val'])

        self.coco_train = COCO(self.ann_file_train)
        self.coco_test = COCO(self.ann_file_val)

        self.categories = categories

        self.idsImgs = []

    def build_dataset_splits(self):
        idsImgs = {}
        for category in self.categories:
            idCatTrainVal = self.coco_train.getCatIds(catNms=category)
            idCatTest = self.coco_test.getCatIds(catNms=category)
            idsImgsTrainVal = self.coco_train.getImgIds(catIds=idCatTrainVal)
            idsImgsTest = self.coco_test.getImgIds(catIds=idCatTest)

            random.shuffle(idsImgsTrainVal)

            idsImgs[category] = {
                "train": idsImgsTrainVal[:self.num_images['train']],
                "val": idsImgsTrainVal[self.num_images['train']: self.num_images['train'] + self.num_images['val']],
                "test": idsImgsTest
            }

        self.idsImgs = idsImgs

    def list_category_names(self):
        cats = self.coco_train.loadCats(self.coco_train.getCatIds())
        return [cat['name'] for cat in cats]

    def download_image(self, img_info):
        response = requests.get(img_info['coco_url'])
        file = open(f"{self.data_path}/{img_info['stage']}/{img_info['category']}/{img_info['file_name']}", "wb")
        file.write(response.content)
        file.close()

    def get_image(self, image_url):
        try:
            image_numpy = io.imread(image_url)
        except Exception as e:
            print(e)
            image_numpy = None
        return image_numpy

    def save_image(self, path, image):
        im = Image.fromarray(image)
        im.save(path)
