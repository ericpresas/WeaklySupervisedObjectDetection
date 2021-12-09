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

        self.model.fc = nn.Flatten()

    def forward(self, x):
        """x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # erase layers you want
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)"""
        x = self.model(x)
        return x