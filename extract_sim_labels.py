import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import Teacher, FeatureExtractor
from datasets import RegionsDataset
import torch.optim as optim
from tqdm import tqdm
device = ("cuda" if torch.cuda.is_available() else "cpu")
import json
from utils import utils, AverageMeter
import numpy as np

print(f"Using {device}...")

root_dir = 'data/coco'

print("Loading Edge Boxes...")
images_info_train = utils.load_pickle(f"{root_dir}/annotations/edge_boxes_val.pkl")
images_info_val = utils.load_pickle(f"{root_dir}/annotations/edge_boxes_val.pkl")

categories_ids = utils.load_pickle(f"{root_dir}/annotations/categories.pkl")

num_classes = len(categories_ids)

print("Loading Support Vector....")

support_feature_vector = utils.load_pickle(f"{root_dir}/annotations/support_feature_vector.pkl")


def extract_similarity_labels(feature_extractor, data):
    batch_logits = np.empty(shape=(0, num_classes))
    batch_pseudo_labels = []

    sim_labels = np.empty(shape=(0, num_classes))

    for batch_element in data:
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
        features_batch = output.to('cpu').numpy()

        for feature in features_batch:
            cos_sim_classes = np.zeros((1, num_classes))
            for i, category_obj in enumerate(categories_ids):
                categoy_name, id_category = category_obj['name'], category_obj['id']
                feature_class_vector = utils.get_feature_vector_class(support_feature_vector, id_category)
                cos_sim_classes[0, i] = utils.compute_class_cosine_similarity(feature, feature_class_vector)

            sim_labels = np.concatenate((sim_labels, cos_sim_classes), axis=0)

        sim_labels = np.exp(sim_labels)
        s_class = sim_labels / np.sum(sim_labels, axis=1).reshape(-1, 1)
        s_det = sim_labels / np.sum(sim_labels, axis=0).reshape(-1, 1).T

        pseudo_logits = np.sum(np.multiply(s_class, s_det), axis=0)
        pseudo_label = np.argmax(pseudo_logits)
        batch_pseudo_labels.append(pseudo_label)

        batch_logits = np.concatenate((batch_logits, np.expand_dims(pseudo_logits, axis=0)), axis=0)

        # http://www.kasimte.com/2020/02/14/how-does-temperature-affect-softmax-in-machine-learning.html

    batch_logits = torch.from_numpy(batch_logits)
    batch_pseudo_labels_oneHot = utils.convertToOneHot(np.array(batch_pseudo_labels), num_classes=num_classes)

    #batch_logits.float().to(device)
    #batch_pseudo_labels = torch.from_numpy(batch_pseudo_labels)
    #batch_pseudo_labels = batch_pseudo_labels.float().to(device)

    return batch_pseudo_labels_oneHot, batch_pseudo_labels


def process(feature_extractor, loader, stage):
    pseudo_labels = []
    loop = tqdm(loader, unit=" batches", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # For printing the progress bar
    loop.set_description(f'[{stage}] Extracting similarity labels')
    for data_regions, image_info in loop:
        data_regions = data_regions.float().to(device)
        pseudo_label_oneHot, pseudo_label = extract_similarity_labels(feature_extractor, data_regions)
        pseudo_labels.append({
            "pseudo_label": pseudo_label_oneHot,
            **image_info
        })
        id_category = image_info['category_id'].numpy()
        print('\n')
        print(f"Label: {id_category[0]} {image_info['category_name'][0]} - Pseudo: {pseudo_label[0]}")
        if int(pseudo_label[0]) == int(id_category[0]):
            print("Match!!")
        else:
            print("No Match...")

    utils.save_pickle(f"{root_dir}/annotations/pseudo_labels_{stage}.pkl")


if __name__ == "__main__":

    feature_extractor = FeatureExtractor()
    feature_extractor.to(device)

    input_size = 224
    batch_size = 1

    data_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.CenterCrop(input_size),
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

    print("Start train Pseudo Labels extractor...")
    process(feature_extractor, dataloaders_dict['train'], 'TRAIN')
    print("Start val Pseudo Labels extractor...")
    process(feature_extractor, dataloaders_dict['val'], 'VAL')




