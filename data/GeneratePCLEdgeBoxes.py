import json
from pycocotools.coco import COCO
import cv2 as cv
import numpy as np
from utils import utils
from multiprocessing import Pool
import time
from tqdm import *
from functools import partial


class GeneratePCLEdgeBoxes(object):
    def __init__(self, path_dataset, path_annotations):
        self.path_dataset = path_dataset
        self.path_annotations = path_annotations

    def load_coco_tools(self, stage):
        ann_file = f"{self.path_annotations}/annotations/instances_{stage}2017.json"
        coco_tools = COCO(ann_file)
        return coco_tools

    @staticmethod
    def imap_unordered_bar(func, imgs, stage, n_processes=2):
        p = Pool(n_processes)
        proposals_pcl = {
            "boxes": [],
            "indexes": [],
            "scores": []
        }
        with tqdm(total=len(imgs)) as pbar:
            for i, res in tqdm(enumerate(p.imap_unordered(partial(func, stage=stage), imgs))):
                pbar.update()
                if res is not None:
                    proposals_pcl['boxes'].append(res['boxes'])
                    proposals_pcl['indexes'].append(res['indexes'])
                    proposals_pcl['scores'].append(res['scores'])

        p.close()
        p.join()
        return proposals_pcl

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

    def process_step(self, imgInfo, stage):
        proposals_pcl = None
        path = f"{self.path_dataset}/{stage}2017/{imgInfo['file_name']}"
        edge_boxes = self.edge_boxes(path)
        if edge_boxes is not None:
            obj_img = {
                "id": imgInfo['id'],
                "edge_boxes": [list(bbox) for bbox in edge_boxes],
            }

            proposals_pcl = {
                "boxes": np.array(obj_img['edge_boxes']),
                "indexes": obj_img['id'],
                "scores": np.array([1.0 for _ in range(len(obj_img['edge_boxes']))])
            }

        return proposals_pcl

    def process(self, stage):
        coco = self.load_coco_tools(stage)


        idsImgs = coco.getImgIds()
        imgsInfo = coco.loadImgs(idsImgs)

        proposals_pcl = self.imap_unordered_bar(self.process_step, imgsInfo, stage, n_processes=6)

        return proposals_pcl



