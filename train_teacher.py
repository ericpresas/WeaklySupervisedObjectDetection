import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import Teacher, FeatureExtractor
from datasets import ClassificationDataset
import torch.optim as optim
from tqdm import tqdm
device = ("cuda" if torch.cuda.is_available() else "cpu")
import json
from utils import utils, AverageMeter
import numpy as np

print(f"Using {device}...")

root_dir = 'data/coco'

categories_ids = utils.load_pickle(f"{root_dir}/annotations/categories.pkl")

num_classes = len(categories_ids)

images_info = utils.load_pickle(f"{root_dir}/annotations/pseudo_labels_train.pkl")

support_images_info = utils.load_pickle(f"{root_dir}/annotations/support_boxes.pkl")


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
        train_loop = tqdm(train_loader, unit=" batches", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # For printing the progress bar
        for data, target in train_loop:
            train_loop.set_description(f'[TRAIN] Epoch {epoch + 1}/{epochs}')
            data, target = data.float().to(device), target.float().to(device)

            features = feature_extractor(data)
            features = features.float().to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=len(target))
            pred = output.round()  # get the prediction
            acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
            train_accuracy.update(acc, n=len(target))
            train_loop.set_postfix(loss=train_loss.avg, accuracy=train_accuracy.avg)

        train_losses.append(train_loss.avg)
        train_accuracies.append(train_accuracy.avg)

        # validation
        model.eval()
        val_loss.reset()
        val_accuracy.reset()
        val_loop = tqdm(val_loader, unit=" batches", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # For printing the progress bar
        with torch.no_grad():
            for data, target in val_loop:
                val_loop.set_description(f'[VAL] Epoch {epoch + 1}/{epochs}')
                data, target = data.float().to(device), target.float().to(device)

                features = feature_extractor(data)
                output = model(features)
                loss = loss_fn(output, target)

                val_loss.update(loss.item(), n=len(target))
                pred = output.round()  # get the prediction
                acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
                val_accuracy.update(acc, n=len(target))
                val_loop.set_postfix(loss=val_loss.avg, accuracy=val_accuracy.avg)

        val_losses.append(val_loss.avg)
        val_accuracies.append(val_accuracy.avg)

    return train_accuracies, train_losses, val_accuracies, val_losses


if __name__ == "__main__":

    feature_extractor = FeatureExtractor()
    teacher_model = Teacher(num_classes)

    feature_extractor.to(device)
    teacher_model.to(device)

    input_size = 224
    batch_size = 32

    data_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {
        "train": ClassificationDataset(dataset_images_info=images_info,
                                       categories_ids=categories_ids,
                                       root_dir=root_dir,
                                       support_images=support_images_info,
                                       transform=data_transforms),
        "val": ClassificationDataset(dataset_images_info=images_info,
                                     categories_ids=categories_ids,
                                     root_dir=root_dir,
                                     transform=data_transforms)
    }

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in
        ['train', 'val']}

    optimizer = optim.Adam(teacher_model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    epochs = 30

    output_train = train_model(teacher_model, feature_extractor, optimizer, loss_fn, dataloaders_dict['train'], dataloaders_dict['val'], epochs)
    train_accuracies, train_losses, val_accuracies, val_losses = output_train




