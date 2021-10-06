import torch.nn as nn
import torch
import torchvision.models as models
#https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet152(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential()

    def forward(self, x):
        x = self.model(x)
        return x

