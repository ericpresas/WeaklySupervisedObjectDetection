import fiftyone as fo
from utils import utils
from pycocotools.coco import COCO


root_dir_results = '/mnt/gpid08/users/eric.presas'
stage = 'val'
root_dir = 'data/coco'
annotations = f"{root_dir}/annotations/instances_{stage}2017.json"
categories_ids = utils.load_pickle(f"{root_dir}/annotations/categories.pkl")
categories_map = {cat_obj['name']:cat_obj['id'] for cat_obj in categories_ids}
categories_map_inverse = {cat_obj['id']:cat_obj['name'] for cat_obj in categories_ids}

export_dataset_dir = f"{root_dir_results}/annotations_NSOD/fake_coco_{stage}"

coco = COCO(annotations)

dataset = fo.Dataset(name="coco-fake-dataset")

if __name__ == "__main__":
    pseudo_labels = utils.load_pickle(f"{root_dir_results}/annotations_NSOD/teacher_predictions{stage}.pkl")

    for instance_label in pseudo_labels:
        if len(instance_label['categories_preds']) > 0:
            sample = fo.Sample(filepath=instance_label['path'])
            catIds = [categories_map[category] for category in instance_label['categories_preds']]

            annotationsIds = coco.getAnnIds(imgIds=[instance_label['id']], catIds=catIds)
            annotations = coco.loadAnns(ids=annotationsIds)

            detections = []

            for annotation in annotations:
                detections.append(
                    fo.Detection(label=categories_map_inverse[annotation['category_id']], bounding_box=annotation['bbox'])
                )

            sample["ground_truth"] = fo.Detections(detections=detections)

            dataset.add_sample(sample)

    label_field = "ground_truth"

    # Export the dataset
    dataset.export(
        export_dir=export_dataset_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field=label_field
    )