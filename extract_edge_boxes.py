from data import GenerateEdgeBoxes

DATASET_IMAGES_PATH = "/mnt/gpid08/datasets/coco-2017/coco/images"
DATASET_ANNOTATIONS_PATH = "/mnt/gpid07/imatge/eric.presas/WeaklySupervisedObjectDetection/data/coco"
if __name__ == "__main__":
    edge_boxes_generator = GenerateEdgeBoxes(path_dataset=DATASET_IMAGES_PATH, path_annotations=DATASET_ANNOTATIONS_PATH)

    edge_boxes_generator.process_stage(stage='train', max_images=500)
    #edge_boxes_generator.process_stage(stage='val', max_images=100)
