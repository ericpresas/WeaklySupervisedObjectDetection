import torch.nn as nn
import torchvision.models as models
import torch


class Teacher(models.resnet.ResNet):
    def __init__(self, num_classes, pretrained=False):
        super(Teacher, self).__init__(models.resnet.Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(models.resnet50(pretrained=True).state_dict())

        self.set_parameter_requires_grad(pretrained)
        num_ftrs = self.fc.in_features
        self.fc = nn.Sequential(
            #nn.ReLU(inplace=True),
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax()
        )

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for name, param in self.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False
