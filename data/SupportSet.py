from pycocotools.coco import COCO
from tqdm import tqdm
from utils import utils
import random


class SupportSet(object):
    def __init__(self, path_dataset, path_annotations):
        self.path_dataset = path_dataset
        self.path_annotations = path_annotations

    def load_coco_tools(self):
        ann_file = f"{self.path_annotations}/annotations/instances_train2017.json"
        coco_tools = COCO(ann_file)
        return coco_tools

    def list_category_names(self, coco_tools, max_classes=20):
        cats = coco_tools.loadCats(coco_tools.getCatIds())
        category_names = [cat['name'] for cat in cats][:max_classes]
        utils.save_pickle(cats[:max_classes], f"{self.path_annotations}/annotations/categories.pkl")
        return category_names

    def process(self, k=20):
        coco = self.load_coco_tools()
        support_set = []
        used_ids = []
        for category in self.list_category_names(coco_tools=coco):
            idCat = coco.getCatIds(catNms=category)
            idsImgs = coco.getImgIds(catIds=idCat)
            imgsInfo = coco.loadImgs(idsImgs)
            annotationsIds = coco.getAnnIds(imgIds=idsImgs)
            annotationsInfo = coco.loadAnns(annotationsIds)
            annotationsInfo = list(filter(lambda x: x['category_id'] == idCat[0], annotationsInfo))

            annotations_gt_area = list(
                filter(lambda x: x['area'] > 5000, annotationsInfo))

            random.shuffle(annotations_gt_area)

            pbar = tqdm(annotations_gt_area[:k], bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
            for annotation in pbar:
                pbar.set_description(f"[Support Set] - {category}")
                imgInfo = list(filter(lambda x: x['id'] == annotation['image_id'], imgsInfo))[0]
                path = f"{self.path_dataset}/train2017/{imgInfo['file_name']}"
                obj_img = {
                    "path": path,
                    "filename": imgInfo['file_name'],
                    "id": imgInfo['id'],
                    "box": [int(point) for point in annotation['bbox']],
                    "category_name": category,
                    "category_id": idCat[0]
                }
                support_set.append(obj_img)
                used_ids.append(imgInfo['id'])

        utils.save_pickle(support_set, path=f"{self.path_annotations}/annotations/support_boxes.pkl")
        utils.save_pickle(used_ids, path=f"{self.path_annotations}/annotations/used_ids.pkl")



