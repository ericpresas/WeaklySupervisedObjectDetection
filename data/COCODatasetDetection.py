from .COCODatasetBase import COCODatasetBase, MaxValueDatasetReached, MaxImagesDatasetReached
import os
from tqdm import tqdm
import json
from utils import utils
import cv2 as cv
import numpy as np


class COCODatasetDetection(COCODatasetBase):
    def __init__(self, data_path: str, data_train_id: int, data_val_id: int, categories: list, max_samples_category: int=50, percentages_dataset: dict=None):
        super().__init__(data_path, data_train_id, data_val_id, categories, max_samples_category, percentages_dataset)
        self.extension_path = 'detection'

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

        with open(f"{data_path}/support/used_images.json") as f:
            self.used_ids = json.load(f)

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
            imgIds = list(filter(lambda x: x not in self.used_ids, imgIds))
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
                            filter(lambda x: x['image_id'] == img_info['id'], annotationsInfo))
                        path_image = f"{self.data_path}/{self.extension_path}/{split}/{category}/{cont}.jpg"
                        edge_boxes = self.fake_annotations(image)
                        for bbox in edge_boxes:
                            self.images_to_save[split][category].append({
                                "path": path_image,
                                #"annotations": [annotation['bbox'] for annotation in annotations_image],
                                "fake_annotations": list(bbox)
                            })
                        self.save_image(path=path_image, image=image)
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

        utils.save_pickle(self.images_to_save, path=f"{self.data_path}/{self.extension_path}/saved_images.pickle")

    @staticmethod
    def fake_annotations(im):
        model = 'resources/model.yml.gz'

        edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
        rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

        orimap = edge_detection.computeOrientation(edges)
        edges = edge_detection.edgesNms(edges, orimap)

        edge_boxes = cv.ximgproc.createEdgeBoxes()
        edge_boxes.setMaxBoxes(2000)
        boxes = edge_boxes.getBoundingBoxes(edges, orimap)

        if len(boxes) > 0:
            return boxes[0]
        else:
            return None