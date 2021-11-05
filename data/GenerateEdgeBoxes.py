import json
from pycocotools.coco import COCO
from tqdm import tqdm
import cv2 as cv
import numpy as np
from utils import utils


class GenerateEdgeBoxes(object):
    def __init__(self, path_dataset, path_annotations):
        self.path_dataset = path_dataset
        self.path_annotations = path_annotations

        self.used_ids = utils.load_pickle(f"{self.path_annotations}/annotations/used_ids.pkl")

        self.categories_ids = utils.load_pickle(f"{self.path_annotations}/annotations/categories.pkl")

    def load_coco_tools(self, stage):
        ann_file = f"{self.path_annotations}/annotations/instances_{stage}2017.json"
        coco_tools = COCO(ann_file)
        return coco_tools

    @staticmethod
    def edge_boxes(path):
        model = '/mnt/gpid07/imatge/eric.presas/WeaklySupervisedObjectDetection/resources/model.yml.gz'

        edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
        im = cv.imread(path)
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

    def process_stage(self, stage, max_images=1000):
        coco = self.load_coco_tools(stage)
        stage_file = []
        boxes = []
        indexes = []
        scores = []
        for category_obj in self.categories_ids:
            category = category_obj['name']
            idCat = coco.getCatIds(catNms=category)
            idsImgs = coco.getImgIds(catIds=idCat)
            idsImgs = list(filter(lambda x: x not in self.used_ids, idsImgs))
            if len(idsImgs) > max_images:
                idsImgs = idsImgs[:max_images]
            imgsInfo = coco.loadImgs(idsImgs)
            pbar = tqdm(imgsInfo, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
            save_info_cat = {
                "stage_file_cat": [],
                "boxes_cat": [],
                "indexes_cat": [],
                "scores_cat": []
            }
            for imgInfo in pbar:
                pbar.set_description(f"[{stage}] - {category}")
                path = f"{self.path_dataset}/{stage}2017/{imgInfo['file_name']}"
                edge_boxes = self.edge_boxes(path)
                if edge_boxes is not None:
                    obj_img = {
                        "path": path,
                        "filename": imgInfo['file_name'],
                        "id": imgInfo['id'],
                        "edge_boxes": [list(bbox) for bbox in edge_boxes],
                        "category_name": category,
                        "category_id": idCat[0]
                    }
                    stage_file.append(obj_img)
                    boxes.append(np.array(obj_img['edge_boxes']))
                    indexes.append(obj_img['id'])
                    scores.append(np.array([1.0 for _ in range(len(obj_img['edge_boxes']))]))

            utils.save_pickle(stage_file, path=f"{self.path_annotations}/annotations/edge_boxes_{stage}.pkl")

        proposals_pcl = {
            "boxes": boxes,
            "indexes": indexes,
            "scores": scores
        }

        utils.save_pickle(proposals_pcl, path=f"{self.path_annotations}/annotations/proposals_pcl_{stage}.pkl")



