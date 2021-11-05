from data import SupportSet

DATASET_IMAGES_PATH = "/mnt/gpid08/datasets/coco-2017/coco/images"
DATASET_ANNOTATIONS_PATH = "/mnt/gpid07/imatge/eric.presas/WeaklySupervisedObjectDetection/data/coco"
if __name__ == "__main__":
    support_set = SupportSet(path_dataset=DATASET_IMAGES_PATH, path_annotations=DATASET_ANNOTATIONS_PATH)

    support_set.process()

