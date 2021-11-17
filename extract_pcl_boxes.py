from data import GeneratePCLEdgeBoxes
from utils import utils

DATASET_IMAGES_PATH = "/mnt/gpid08/datasets/coco-2017/coco/images"
DATASET_ANNOTATIONS_PATH = "/mnt/gpid07/imatge/eric.presas/WeaklySupervisedObjectDetection/data/coco"
if __name__ == "__main__":
    edge_boxes_generator = GeneratePCLEdgeBoxes(path_dataset=DATASET_IMAGES_PATH, path_annotations=DATASET_ANNOTATIONS_PATH)

    pcl_boxes_train = edge_boxes_generator.process(stage='train')
    pcl_boxes_val = edge_boxes_generator.process(stage='val')

    pcl_boxes = {
        "boxes": pcl_boxes_train['boxes'] + pcl_boxes_val['boxes'],
        "indexes": pcl_boxes_train['indexes'] + pcl_boxes_val['indexes'],
        "scores": pcl_boxes_train['scores'] + pcl_boxes_val['scores']
    }

    utils.save_pickle(pcl_boxes, path=f"{DATASET_ANNOTATIONS_PATH}/annotations/proposals_pcl_trainval.pkl")
    utils.save_pickle(pcl_boxes_val, path=f"{DATASET_ANNOTATIONS_PATH}/annotations/coco_2017_val.pkl")