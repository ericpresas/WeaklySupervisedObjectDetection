from .COCODatasetBasev2 import COCODatasetBase, MaxValueDatasetReached, MaxImagesDatasetReached
import os
from tqdm import tqdm
import json
import random


class COCODatasetClassification(COCODatasetBase):
    def __init__(self, data_path: str, data_train_id: int, data_val_id: int, categories: list, max_samples_category: int=20, percentages_dataset: dict=None):
        super().__init__(data_path, data_train_id, data_val_id, categories, max_samples_category, percentages_dataset)
        self.extension_path = 'support'

        directory_exists = os.path.exists(f"{self.data_path}/{self.extension_path}")
        if not directory_exists:
            os.mkdir(f"{self.data_path}/{self.extension_path}")

        exists_file_images = os.path.exists(f"{self.data_path}/{self.extension_path}/saved_images.json")

        self.images_to_save = {
            "train": {},
            "val": {},
            "test": {}
        }

        if exists_file_images:
            with open(f"{self.data_path}/{self.extension_path}/saved_images.json") as f:
                self.images_to_save = json.load(f)

    def save_images(self, split):

        # Save categories ids
        categories_ids = {category: id for id, category in enumerate(self.categories)}
        if split == 'train':
            with open(f"{self.data_path}/{self.extension_path}/categories.json", 'w') as outfile:
                json.dump(categories_ids, outfile, indent=4)

        directory_train_exists = os.path.exists(f"{self.data_path}/{self.extension_path}/{split}")
        if not directory_train_exists:
            os.mkdir(f"{self.data_path}/{self.extension_path}/{split}")
        coco = self.coco_train
        if split == 'test':
            coco = self.coco_test
        for category, imgIds in self.idsImgs.items():
            imgIds = imgIds[split]
            catId = self.coco_train.getCatIds(catNms=category)
            catId = catId[0]
            annotationsIds = coco.getAnnIds(imgIds=imgIds)
            imgsInfo = coco.loadImgs(imgIds)
            annotationsInfo = coco.loadAnns(annotationsIds)

            annotationsInfo = list(filter(lambda x: x['category_id'] == catId, annotationsInfo))

            directory_exists = os.path.exists(f"{self.data_path}/{self.extension_path}/{split}/{category}")
            if not directory_exists:
                os.mkdir(f"{self.data_path}/{self.extension_path}/{split}/{category}")

            cont = 0
            self.images_to_save[split][category] = []

            try:
                pbar = tqdm(total=self.num_images[split])
                for img_info in imgsInfo:
                    image = self.get_image(img_info['coco_url'])

                    if image is not None:
                        annotations_image = list(
                            filter(lambda x: x['image_id'] == img_info['id'] and x['area'] > 5000, annotationsInfo))
                        for annotation in annotations_image:
                            path_image = f"{self.data_path}/{self.extension_path}/{split}/{category}/{cont}.jpg"
                            x, y, w, h = [int(point) for point in annotation['bbox']]
                            cropped_image = image[y:y + h, x:x + w]

                            self.images_to_save[split][category].append(path_image)
                            self.save_image(path=path_image, image=cropped_image)
                            cont += 1
                            pbar.update(1)
                            if cont == self.num_images[split]:
                                pbar.close()
                                raise MaxValueDatasetReached

                raise MaxImagesDatasetReached

            except MaxImagesDatasetReached as e:
                num_bar_left = self.num_images[split] - cont
                for i in range(num_bar_left):
                    pbar.update(1)
                pbar.close()
                print(f'Not enough images for {split} {category}')

            except MaxValueDatasetReached as e:
                print(f'Maximum value Reached for {split} {category}')

        with open(f"{self.data_path}/{self.extension_path}/saved_images.json", 'w') as outfile:
            json.dump(self.images_to_save, outfile, indent=4)