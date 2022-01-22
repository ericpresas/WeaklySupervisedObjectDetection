import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import Teacher
from datasets import RegionsDataset
from tqdm import tqdm
device = ("cuda" if torch.cuda.is_available() else "cpu")
from utils import utils, APMeter
import numpy as np
import torch.nn as nn
from pycocotools.coco import COCO
from config import config

torch.cuda.empty_cache()

print(f"Using {device}...")

stage = 'val'

apmeter = APMeter()
root_dir = 'data/coco'
root_dir_results = '/mnt/gpid08/users/eric.presas'
checkpoint_path = 'outputs/model-2021-12-01_19:49:46.pt'

print("Loading Edge Boxes...")
#images_info_train = utils.load_pickle(f"{root_dir}/annotations/edge_boxes_train.pkl")
images_info = utils.load_pickle(f"{root_dir}/annotations/edge_boxes_{stage}.pkl")

#annotations_train = f"{root_dir}/annotations/instances_train2017.json"
annotations = f"{root_dir}/annotations/instances_{stage}2017.json"

#coco_train = COCO(annotations_train)
coco = COCO(annotations)

categories_ids = utils.load_pickle(f"{root_dir}/annotations/categories.pkl")

num_classes = len(categories_ids)

#pseudo_labels_train = utils.load_pickle(f"{root_dir}/annotations/pseudo_labels_train.pkl")
pseudo_labels = utils.load_pickle(f"{root_dir}/annotations/pseudo_labels_{stage}.pkl")

support_images_info = utils.load_pickle(f"{root_dir}/annotations/support_boxes.pkl")

softmax = nn.Softmax(dim=1)


def extract_gt(classes):
    array_multiclass = np.array([1 if category['name'] in classes else 0 for category in categories_ids])
    return torch.from_numpy(array_multiclass)


def from_preds_to_categories(pred, treshold=0.4):
    pred_array = torch.squeeze(pred).numpy()
    indexes = np.where(pred_array >= treshold)
    categories_preds = [categories_ids[index]['name'] for index in indexes[0]]
    return categories_preds


def print_stats(average_precision):
    print('------- AP Classes ----------')
    array_precision = average_precision.numpy()
    for i, category in enumerate(categories_ids):
        print(f"{category['name']}: {array_precision[i]}")

    print(f"All: {np.mean(array_precision)}")


def process(teacher_model, loader, stage, images_info_stage, pseudo_labels):
    loop = tqdm(loader, unit=" batches", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # For printing the progress bar
    loop.set_description(f'[{stage}] Extracting multiclass labels')

    stage_preds = []

    count = 0
    with torch.no_grad():
        for data_regions, image_info in loop:
            if data_regions is not None:
                id = int(image_info['id'].numpy())
                img_obj = list(filter(lambda x: x['id'] == id, images_info_stage))

                if len(img_obj) > 0:
                    count += 1
                    img_obj = img_obj[0]
                    sim_obj = list(filter(lambda x: x['id'] == id, pseudo_labels))
                    if len(sim_obj) > 0:
                        sim_obj = sim_obj[0]
                        comb_preds = []
                        sim_preds = sim_obj['sim_logits']
                        comb_preds.append(sim_preds)
                        data_regions = data_regions.float().to(device)
                        data_regions = torch.squeeze(data_regions)
                        try:
                            output = teacher_model(data_regions)
                            preds_regions = softmax(output)
                            preds = torch.unsqueeze(torch.mean(preds_regions, dim=0), dim=0)
                            preds = preds.to('cpu')
                            comb_preds.append(preds)

                            comb_preds = torch.stack(comb_preds)
                            comb_preds = torch.mean(comb_preds, dim=0)

                            thr_preds = from_preds_to_categories(comb_preds, treshold=0.8)

                            stage_preds.append({
                                "id": img_obj['id'],
                                "path": img_obj['path'],
                                "categories_preds": thr_preds
                            })

                            ground_truth = torch.unsqueeze(extract_gt(sim_obj['categories']), dim=0)
                            apmeter.add(output=comb_preds, target=ground_truth)

                            if count % 10 == 0:
                                average_precision = apmeter.value()
                                print_stats(average_precision)
                                """utils.save_pickle(stage_preds,
                                                  f"{root_dir_results}/annotations_NSOD/teacher_predictions{stage}.pkl")"""
                        except Exception as e:
                            print('Error')
                            print(e)


if __name__ == "__main__":
    teacher_model = Teacher(num_classes)
    teacher_model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    teacher_model.load_state_dict(checkpoint['model_state_dict'])

    teacher_model.eval()

    input_size = 224
    batch_size = 1

    data_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_dataset = RegionsDataset(dataset_images_info=images_info, categories_ids=categories_ids, root_dir=root_dir,
                              transform=data_transforms)

    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"Start {stage} Multiclass Labels extractor...")
    process(teacher_model, dataloader, stage, images_info, pseudo_labels)

    average_precision = apmeter.value()
    print_stats(average_precision)

