import torch
import torch.nn as nn
import torch.nn.functional as F
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

root_dir = 'data/coco'

images_info = utils.load_pickle(f"{root_dir}/detection/saved_images.pickle")

with open(f"{root_dir}/support/categories.json") as f:
    categories_ids = json.load(f)

num_classes = len(categories_ids.keys())

support_feature_vector = utils.load_pickle(f"{root_dir}/support/support_feature_vector.pickle")

def extract_similarity_labels(feature_extractor, data):
    batch_logits = np.empty(shape=(0, num_classes))

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
            for categoy_name, id_category in categories_ids.items():
                feature_class_vector = utils.get_feature_vector_class(support_feature_vector, id_category)
                cos_sim_classes[0, id_category] = utils.compute_class_cosine_similarity(feature, feature_class_vector)

            sim_labels = np.concatenate((sim_labels, cos_sim_classes), axis=0)

        sim_labels = np.exp(sim_labels)
        s_class = sim_labels / np.sum(sim_labels, axis=1).reshape(-1, 1)
        s_det = sim_labels / np.sum(sim_labels, axis=0).reshape(-1, 1).T

        pseudo_labels = np.sum(np.multiply(s_class, s_det), axis=0)

        batch_logits = np.concatenate((batch_logits, np.expand_dims(pseudo_labels, axis=0)), axis=0)

        # http://www.kasimte.com/2020/02/14/how-does-temperature-affect-softmax-in-machine-learning.html

    batch_logits = torch.from_numpy(batch_logits)

    batch_logits.float().to(device)

    return batch_logits

def train_model(model, feature_extractor, optimizer, loss_fn, train_loader, val_loader, epochs):
    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()

    for epoch in range(epochs):
        model.train()
        train_loss.reset()
        train_accuracy.reset()
        train_loop = tqdm(train_loader, unit=" batches")  # For printing the progress bar
        for data, target in train_loop:
            train_loop.set_description('[TRAIN] Epoch {}/{}'.format(epoch + 1, epochs))
            data, target = data.float().to(device), target.float().to(device)
            data_logits = extract_similarity_labels(feature_extractor, data)
            print(data_logits.shape)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, data_logits)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=len(data_logits))
            pred = output.round()  # get the prediction
            acc = pred.eq(target.view_as(pred)).sum().item() / len(data_logits)
            train_accuracy.update(acc, n=len(data_logits))
            train_loop.set_postfix(loss=train_loss.avg, accuracy=train_accuracy.avg)

if __name__ == "__main__":

    feature_extractor = FeatureExtractor()
    teacher_model = Teacher(num_classes)

    feature_extractor.to(device)
    teacher_model.to(device)

    input_size = 224
    batch_size = 2

    data_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {
        "train": RegionsDataset(dataset_images_info=images_info, categories_ids=categories_ids, root_dir=root_dir, type='train', level='image', transform=data_transforms),
        "val": RegionsDataset(dataset_images_info=images_info, categories_ids=categories_ids, root_dir=root_dir, type='val', level='image', transform=data_transforms)
    }

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=0) for x in
        ['train', 'val']}

    optimizer = optim.Adam(teacher_model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    epochs = 30

    train_model(teacher_model, feature_extractor, optimizer, loss_fn, dataloaders_dict['train'], dataloaders_dict['val'], epochs)





