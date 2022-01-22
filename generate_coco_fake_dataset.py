import os
os.system('pkill mongod')

import fiftyone as fo
from utils import utils
from pycocotools.coco import COCO
from PIL import Image


root_dir_results = '/mnt/gpid08/users/eric.presas'
stage = 'val'
root_dir = 'data/coco'
annotations_info = f"{root_dir}/annotations/instances_{stage}2017.json"
categories_ids = utils.load_pickle(f"{root_dir}/annotations/categories.pkl")
categories_map = {cat_obj['name']:cat_obj['id'] for cat_obj in categories_ids}
categories_map_inverse = {cat_obj['id']:cat_obj['name'] for cat_obj in categories_ids}

images_path = f"/mnt/gpid08/datasets/coco-2017/coco/images/{stage}2017"

export_dataset_dir = f"{root_dir_results}/annotations_NSOD/fake_coco_{stage}"

coco = COCO(annotations_info)
try:
    fo.delete_dataset(name=f"coco-fake-dataset-{stage}")
except Exception as e:
    print(f"coco-fake-dataset-{stage} does not exist!")
dataset = fo.Dataset(name=f"coco-fake-dataset-{stage}")

if __name__ == "__main__":
    pseudo_labels = utils.load_pickle(f"{root_dir_results}/annotations_NSOD/teacher_predictions{stage}.pkl")

    for instance_label in pseudo_labels:
        if len(instance_label['categories_preds']) > 0:
            sample = fo.Sample(filepath=instance_label['path'])
            img = Image.open(instance_label['path']).convert("RGB")

            width, height = img.size

            catIds = [categories_map[category] for category in instance_label['categories_preds']]

            annotationsIds = coco.getAnnIds(imgIds=[instance_label['id']], catIds=catIds)
            annotations = coco.loadAnns(ids=annotationsIds)

            detections = []

            for annotation in annotations:
                bbox = [annotation['bbox'][0]/width, annotation['bbox'][1]/height, annotation['bbox'][2]/width, annotation['bbox'][3]/height]
                detections.append(
                    fo.Detection(label=categories_map_inverse[annotation['category_id']], bounding_box=bbox)
                )

            if len(detections) > 0:
                sample["ground_truth"] = fo.Detections(detections=detections)
                sample['coco_id'] = instance_label['id']

                dataset.add_sample(sample)

    label_field = "ground_truth"

    # Export the dataset
    dataset.export(
        export_dir=export_dataset_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field=label_field,

    )