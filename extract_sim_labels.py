import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import FeatureExtractor
from datasets import RegionsDataset
from tqdm import tqdm
device = ("cuda" if torch.cuda.is_available() else "cpu")
from utils import utils
import numpy as np
from pycocotools.coco import COCO
from config import config

print(f"Using {device}...")

root_dir = 'data/coco'

print("Loading Edge Boxes...")
images_info_train = utils.load_pickle(f"{root_dir}/annotations/edge_boxes_train.pkl")
images_info_val = utils.load_pickle(f"{root_dir}/annotations/edge_boxes_val.pkl")

annotations_train = f"{root_dir}/annotations/instances_train2017.json"
annotations_val = f"{root_dir}/annotations/instances_val2017.json"

coco_train = COCO(annotations_train)
coco_val = COCO(annotations_val)

categories_ids = utils.load_pickle(f"{root_dir}/annotations/categories.pkl")

num_classes = len(categories_ids)

print("Loading Support Vector....")

support_feature_vector = utils.load_pickle(f"{root_dir}/annotations/support_feature_vector.pkl")
support_tensor = torch.from_numpy(support_feature_vector['features'])


def get_categories_image(image_id, stage):
    if stage == 'train':
        coco = coco_train
    else:
        coco = coco_val

    idsCats = [cat['id'] for cat in categories_ids]
    idsMapCats = {cat['id']:cat['name'] for cat in categories_ids}
    annIds = coco.getAnnIds([image_id], idsCats)
    anns = coco.loadAnns(annIds)
    return set([idsMapCats[ann['category_id']] for ann in anns])


def compute_softmax(sim_labels, axis=0):
    softmax = []
    for i in range(sim_labels.shape[axis]):
        if axis == 1:
            vector = sim_labels[:, i]
        else:
            vector = sim_labels[i, :]
        logits = utils.logitsFrom(vector)
        low_temp = 0.5
        logits_low_temp = [x / low_temp for x in logits]

        softmax.append(np.array(utils.softmax(logits_low_temp)))

    if axis == 1:
        return np.transpose(np.stack(softmax))
    else:
        return np.stack(softmax)


def compute_softmax_torch(sim_labels, axis=0):
    softmax_values = []
    for i in range(sim_labels.shape[axis]):
        if axis == 1:
            vector = sim_labels[:, i]
        else:
            vector = sim_labels[i, :]
        logits = torch.special.logit(vector)
        low_temp = 0.1
        logits_low_temp = torch.div(logits, low_temp)

        softmax = torch.nn.Softmax(dim=0)

        softmax_values.append(softmax(logits_low_temp))

    if axis == 1:
        return torch.transpose(torch.stack(softmax_values), 0, 1)
    else:
        return torch.stack(softmax_values)


def extract_similarity_labels(feature_extractor, data):
    batch_logits = []
    batch_pseudo_labels = []

    sim_labels = []
    for batch_element in data:
        batch_element = batch_element.to(device)
        output = feature_extractor(batch_element)
        """chunked_tensor = torch.chunk(batch_element, 1)
        output = None
        for i, chunk in enumerate(chunked_tensor):
            if i == 0:
                output = feature_extractor(chunk)
            else:
                features = feature_extractor(chunk)
                output = torch.cat((output, features), 0)"""
        # print(output.shape)
        features_batch = output
        for feature in features_batch:
            cos_sim_classes = torch.zeros(num_classes).to(device)
            cos_sim_classes_v2 = utils.compute_cosine_sim_tensors(feature.to(device), support_tensor.to(device))
            for i, category_obj in enumerate(categories_ids):
                categoy_name, id_category = category_obj['name'], category_obj['id']
                similarities_class_instances = utils.get_values_vector_class(cos_sim_classes_v2, support_feature_vector['labels'], id_category)
                cos_sim_classes[i] = torch.mean(similarities_class_instances)

            sim_labels.append(cos_sim_classes)

        sim_labels = torch.stack(sim_labels)

        #s_class_v2 = compute_softmax(sim_labels.numpy(), axis=0)
        #s_det_v2 = compute_softmax(sim_labels.numpy(), axis=1)

        s_class = compute_softmax_torch(sim_labels, axis=0)
        s_det = compute_softmax_torch(sim_labels, axis=1)

        pseudo_logits = torch.sum(torch.mul(s_class, s_det), dim=0)

        pseudo_label = torch.argmax(pseudo_logits)
        batch_pseudo_labels.append(pseudo_label)

        batch_logits.append(pseudo_logits)

        # http://www.kasimte.com/2020/02/14/how-does-temperature-affect-softmax-in-machine-learning.html

    batch_logits = torch.stack(batch_logits)
    batch_pseudo_labels = torch.stack(batch_pseudo_labels)
    batch_pseudo_labels_oneHot = utils.convertToOneHot(batch_pseudo_labels.to('cpu').numpy(), num_classes=num_classes)

    return batch_pseudo_labels_oneHot, batch_logits.to('cpu'), batch_pseudo_labels.to('cpu')


def process(feature_extractor, loader, stage, images_info_stage):
    pseudo_labels = []
    loop = tqdm(loader, unit=" batches", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # For printing the progress bar
    loop.set_description(f'[{stage}] Extracting similarity labels')

    matches = 0
    count = 0
    for data_regions, image_info in loop:
        id = int(image_info['id'].numpy())
        img_obj = list(filter(lambda x: x['id'] == id, images_info_stage))

        if len(img_obj) > 0:
            img_obj = img_obj[0]
            count += 1
            data_regions = data_regions.float().to(device)
            pseudo_label_oneHot, logits, pseudo_label = extract_similarity_labels(feature_extractor, data_regions)
            categories_image = get_categories_image(img_obj['id'], stage)
            pseudo_label_cat = categories_ids[int(pseudo_label[0])]
            pseudo_labels.append({
                "pseudo_label_id": pseudo_label_cat['id'],
                "pseudo_label": pseudo_label_oneHot,
                "sim_logits": logits,
                "id": img_obj['id'],
                "categories": list(categories_image),
                "path": img_obj['path']
            })

            if pseudo_label_cat['name'] in categories_image:
                matches += 1

            loop.set_postfix({"Accuracy": matches/count})

            #utils.save_pickle(pseudo_labels, f"{root_dir}/annotations/pseudo_labels_{stage}.pkl")
        else:
            loop.set_postfix({"Error: ": "No image"})

if __name__ == "__main__":

    feature_extractor = FeatureExtractor()
    feature_extractor.to(device)

    # SOFTKNN, HARDKNN

    feature_extractor.eval()

    input_size = 224
    batch_size = 1

    data_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {
        "train": RegionsDataset(dataset_images_info=images_info_train, categories_ids=categories_ids, root_dir=root_dir, transform=data_transforms),
        "val": RegionsDataset(dataset_images_info=images_info_val, categories_ids=categories_ids, root_dir=root_dir, transform=data_transforms)
    }

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in
        ['train', 'val']}

    #print("Start train Pseudo Labels extractor...")
    #process(feature_extractor, dataloaders_dict['train'], 'train', images_info_train)
    print("Start val Pseudo Labels extractor...")
    process(feature_extractor, dataloaders_dict['val'], 'val', images_info_val)




