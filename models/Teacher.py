import torch.nn as nn
from torch.nn import LogSoftmax
import torchvision.models as models


class Teacher(nn.Module):
    def __init__(self, n_classes):
        super(Teacher, self).__init__()
        original_model = models.resnet152(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-1])

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(2048, n_classes)
        self.drop_layer = nn.Dropout(p=0.5)

        #self.fc1 = nn.Linear(2048, 512)
        #self.fc2 = nn.Linear(512, n_classes)
        #self.fc3 = nn.Linear(2048, n_classes)
        #self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        #x = self.fc1(x)
        #x = self.fc2(x)
        #output = self.fc3(x)
        #output = self.logSoftmax(x)
        return output


