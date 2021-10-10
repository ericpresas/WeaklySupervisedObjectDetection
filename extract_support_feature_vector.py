import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import FeatureExtractor
from datasets import SupportDataset
from tqdm import tqdm
device = ("cuda" if torch.cuda.is_available() else "cpu")
import json
from utils import utils


root_dir = 'data/coco'
with open(f"{root_dir}/support/saved_images.json") as f:
    images_info = json.load(f)


with open(f"{root_dir}/support/categories.json") as f:
    categories_ids = json.load(f)





if __name__ == "__main__":
    feature_extractor = FeatureExtractor()

    #feature_extractor.to(device)

    #feature_extractor.eval()

    input_size = 224
    batch_size = 16

    data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    support_dataset = SupportDataset(dataset_images_info=images_info, categories_ids=categories_ids, root_dir=root_dir, transform=data_transforms)

    support_dataloader = torch.utils.data.DataLoader(support_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    extract_features_loop = tqdm(support_dataloader, unit=" batches")  # For printing the progress bar

    support_feature_vector = np.empty(shape=(0, 2048))
    support_labels = np.empty(shape=0)
    with torch.no_grad():
        for data, target in extract_features_loop:
            output = feature_extractor(data)
            #print(output.shape)
            features = output.to('cpu').numpy()
            labels = target.to('cpu').numpy()
            support_feature_vector = np.concatenate((support_feature_vector, features), axis=0)
            support_labels = np.concatenate((support_labels, labels), axis=0)

    support_features_data = {
        "features": support_feature_vector,
        "labels": support_labels
    }

    utils.save_pickle(support_features_data, path='data/coco/support/support_feature_vector.pickle')


