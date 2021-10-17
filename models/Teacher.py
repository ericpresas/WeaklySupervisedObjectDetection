import torch.nn as nn


class Teacher(nn.Module):
    def __init__(self, n_classes):
        super(Teacher, self).__init__()

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


