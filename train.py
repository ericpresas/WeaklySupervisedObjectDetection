import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import Teacher
from datasets import ClassificationDataset
import torch.optim as optim
from tqdm import tqdm
device = ("cuda" if torch.cuda.is_available() else "cpu")
import json
from utils import AverageMeter

root_dir = 'data/coco'
with open(f"{root_dir}/classification/saved_images.json") as f:
    images_info = json.load(f)


with open(f"{root_dir}/classification/categories.json") as f:
    categories_ids = json.load(f)


def train_model(model, optimizer, loss_fn, train_loader, val_loader, epochs):
    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()

    for epoch in range(epochs):
        # train
        model.train()
        train_loss.reset()
        train_accuracy.reset()
        train_loop = tqdm(train_loader, unit=" batches")  # For printing the progress bar
        for data, target in train_loop:
            train_loop.set_description('[TRAIN] Epoch {}/{}'.format(epoch + 1, epochs))
            data, target = data.float().to(device), target.float().to(device)
            #target = target.unsqueeze(-1)
            optimizer.zero_grad()
            output = model(data)
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
        val_loop = tqdm(val_loader, unit=" batches")  # For printing the progress bar
        with torch.no_grad():
            for data, target in val_loop:
                val_loop.set_description('[VAL] Epoch {}/{}'.format(epoch + 1, epochs))
                data, target = data.float().to(device), target.float().to(device)
                #target = target.unsqueeze(-1)
                output = model(data)
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
    teacher_model = Teacher(num_classes=len(categories_ids.keys()), pretrained=True)

    for name, param in teacher_model.named_parameters():
        print(f"{name} - {param.requires_grad}")

    teacher_model.to(device)

    input_size = 224
    batch_size = 128

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        "train": ClassificationDataset(dataset_images_info=images_info, categories_ids=categories_ids, root_dir=root_dir, type='train', transform=data_transforms['train']),
        "val": ClassificationDataset(dataset_images_info=images_info, categories_ids=categories_ids, root_dir=root_dir, type='val', transform=data_transforms['val'])
    }

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}

    optimizer = optim.Adam(teacher_model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    epochs = 30
    train_accuracies, train_losses, val_accuracies, val_losses = train_model(teacher_model, optimizer, loss_fn,
                                                                             dataloaders_dict['train'], dataloaders_dict['val'], epochs)

