from utils import utils
from itertools import groupby
from PIL import Image
import matplotlib.pyplot as plt
DATASET_IMAGES_PATH = "/mnt/gpid08/datasets/coco-2017/coco/images"
DATASET_ANNOTATIONS_PATH = "/mnt/gpid07/imatge/eric.presas/WeaklySupervisedObjectDetection/data/coco"


if __name__ == "__main__":
    filename = f"{DATASET_ANNOTATIONS_PATH}/annotations/support_boxes.pkl"
    support_boxes = utils.load_pickle(filename)
    for k, v in groupby(support_boxes, key=lambda x: x['category_name']):
        fig = plt.figure(figsize=(10, 7))
        # setting values to rows and column variables
        rows = 4
        columns = 5
        print(k)
        category_images_info = list(v)
        for i, category_image_info in enumerate(category_images_info):
            img = Image.open(category_image_info['path']).convert("RGB")
            x, y, w, h = category_image_info['box']
            cropped_image = img.crop((x, y, x + w, y + h))
            # Adds a subplot at the 1st position
            fig.add_subplot(rows, columns, i+1)

            # showing image
            plt.imshow(cropped_image)
            plt.axis('off')
        fig.suptitle(k, fontsize=16)
        plt.show()

