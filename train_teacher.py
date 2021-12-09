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
import matplotlib.pyplot as plt
from datetime import datetime
now = datetime.now()
date_time_string = now.strftime("%Y-%m-%d_%H:%M:%S")
PATH_MODEL = f"outputs/model-{date_time_string}.pt"

print(f"Using {device}...")

root_dir = 'data/coco'

categories_ids = utils.load_pickle(f"{root_dir}/annotations/categories.pkl")

num_classes = len(categories_ids)

images_info = utils.load_pickle(f"{root_dir}/annotations/pseudo_labels_train.pkl")
images_info_val = utils.load_pickle(f"{root_dir}/annotations/pseudo_labels_val.pkl")

support_images_info = utils.load_pickle(f"{root_dir}/annotations/support_boxes.pkl")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).cpu().detach().numpy())
    return res


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
            target = target.long()

            #features = feature_extractor(data)
            #features = features.float().to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = loss_fn(output, torch.max(target, 1)[1])
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=len(target))
            #pred = output.round()

            acc = accuracy(output, torch.max(target, 1)[1])[0]
            #acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
            train_accuracy.update(acc, n=len(target))
            train_loop.set_postfix(loss=train_loss.avg, accuracy=train_accuracy.avg)

        if len(train_accuracies) > 0:
            if train_accuracy.avg > train_accuracies[-1]:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss.avg
                }, PATH_MODEL)

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
                target = target.long()

                #features = feature_extractor(data)
                output = model(data)
                loss = loss_fn(output, torch.max(target, 1)[1])

                val_loss.update(loss.item(), n=len(target))
                #pred = output.round()  # get the prediction

                acc = accuracy(output, torch.max(target, 1)[1])[0]
                #acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
                val_accuracy.update(acc, n=len(target))
                val_loop.set_postfix(loss=val_loss.avg, accuracy=val_accuracy.avg)

        val_losses.append(val_loss.avg)
        val_accuracies.append(val_accuracy.avg)

        fig, (ax1, ax2) = plt.subplots(2)
        ax1.set_title("Accuracy")
        ax2.set_title("Loss")

        ax1.plot(train_accuracies, color='r', label='Train')
        ax1.plot(val_accuracies, color='g', label='Val')
        ax1.legend()

        ax2.plot(train_losses, color='r', label='Train')
        ax2.plot(val_losses, color='g', label='Val')
        ax2.legend()

        plt.savefig(f'Metrics-{date_time_string}.png')

    return train_accuracies, train_losses, val_accuracies, val_losses


if __name__ == "__main__":

    feature_extractor = FeatureExtractor()
    teacher_model = Teacher(num_classes)

    feature_extractor.to(device)
    teacher_model.to(device)

    feature_extractor.eval()

    input_size = 224
    batch_size = 128

    data_transforms_train = transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms_val =transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {
        "train": ClassificationDataset(dataset_images_info=images_info,
                                       categories_ids=categories_ids,
                                       root_dir=root_dir,
                                       support_images=support_images_info,
                                       transform=data_transforms_train),
        "val": ClassificationDataset(dataset_images_info=images_info_val,
                                     categories_ids=categories_ids,
                                     root_dir=root_dir,
                                     transform=data_transforms_val)
    }

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in
        ['train', 'val']}

    optimizer = optim.Adam(teacher_model.parameters(), lr=1e-3, weight_decay=0.0001)
    #loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.CrossEntropyLoss()

    epochs = 300

    output_train = train_model(teacher_model, feature_extractor, optimizer, loss_fn, dataloaders_dict['train'], dataloaders_dict['val'], epochs)
    train_accuracies, train_losses, val_accuracies, val_losses = output_train




