from pycocotools.coco import COCO
import json
import os
from tqdm import tqdm
from skimage import io
from PIL import Image


class MaxValueDatasetReached(Exception):
    " Raised when max defined value of samples is reached. "
    pass


class MaxImagesDatasetReached(Exception):
    " Raised when no more samples in dataset. "
    pass


class COCOSupportSet(object):
    def __init__(self, data_path: str, data_id: str, categories: list, max_samples_category: int=20):

        self.extension_path = 'support'

        self.data_path = data_path
        self.data_id = data_id

        directory_exists = os.path.exists(f"{self.data_path}/{self.extension_path}")
        if not directory_exists:
            os.mkdir(f"{self.data_path}/{self.extension_path}")

        self.max_samples = max_samples_category

        self.ann_file = f"{self.data_path}/annotations/instances_{self.data_id}.json"

        self.coco = COCO(self.ann_file)

        self.categories = categories

        self.idsImgs = {}

        self.images_to_save = {}

        self.used_ids = []

    def list_category_names(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        return [cat['name'] for cat in cats]

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

    def save_images(self):

        for category in self.categories:
            idCat = self.coco.getCatIds(catNms=category)
            self.idsImgs[category] = self.coco.getImgIds(catIds=idCat)

        # Save categories ids
        categories_ids = {category: id for id, category in enumerate(self.categories)}
        
        with open(f"{self.data_path}/{self.extension_path}/categories.json", 'w') as outfile:
            json.dump(categories_ids, outfile, indent=4)

        directory_train_exists = os.path.exists(f"{self.data_path}/{self.extension_path}/images")
        if not directory_train_exists:
            os.mkdir(f"{self.data_path}/{self.extension_path}/images")

        for category, imgIds in self.idsImgs.items():
            catId = self.coco.getCatIds(catNms=category)
            catId = catId[0]
            annotationsIds = self.coco.getAnnIds(imgIds=imgIds)
            imgsInfo = self.coco.loadImgs(imgIds)
            annotationsInfo = self.coco.loadAnns(annotationsIds)

            annotationsInfo = list(filter(lambda x: x['category_id'] == catId, annotationsInfo))

            directory_exists = os.path.exists(f"{self.data_path}/{self.extension_path}/images/{category}")
            if not directory_exists:
                os.mkdir(f"{self.data_path}/{self.extension_path}/images/{category}")

            cont = 0
            self.images_to_save[category] = []

            pbar = tqdm(total=self.max_samples)
            try:
                for img_info in imgsInfo:
                    image = self.get_image(img_info['coco_url'])

                    if image is not None:
                        annotations_image = list(
                            filter(lambda x: x['image_id'] == img_info['id'] and x['area'] > 5000, annotationsInfo))

                        if len(annotations_image) > 0:
                            self.used_ids.append(img_info['id'])

                        for annotation in annotations_image:
                            path_image = f"{self.data_path}/{self.extension_path}/images/{category}/{cont}.jpg"
                            x, y, w, h = [int(point) for point in annotation['bbox']]
                            cropped_image = image[y:y + h, x:x + w]

                            self.images_to_save[category].append(path_image)
                            self.save_image(path=path_image, image=cropped_image)
                            cont += 1
                            pbar.update(1)
                            if cont == self.max_samples:
                                pbar.close()
                                raise MaxValueDatasetReached

                raise MaxImagesDatasetReached

            except MaxImagesDatasetReached as e:
                num_bar_left = self.max_samples - cont
                for i in range(num_bar_left):
                    pbar.update(1)
                pbar.close()
                print(f'Not enough images for images {category}')

            except MaxValueDatasetReached as e:
                print(f'Maximum value Reached for images {category}')

        with open(f"{self.data_path}/{self.extension_path}/saved_images.json", 'w') as outfile:
            json.dump(self.images_to_save, outfile, indent=4)

        with open(f"{self.data_path}/{self.extension_path}/used_images.json", 'w') as outfile:
            json.dump(self.used_ids, outfile, indent=4)