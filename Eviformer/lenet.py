import torch.nn as nn
import torch.nn.functional as F
import warnings
from collections import OrderedDict

class LeNet(nn.Module):
    def __init__(self, dropout: bool = False, num_classes: int = 4):
        super().__init__()
        self.use_dropout = dropout

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15),  # The first parameter in in_channel may need to be changed, here it is 1
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )


        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),  # 128,8,8
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))  # 128, 4,4

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),

        )
        self.fc = nn.Linear(256, num_classes)


    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc(x)

        return x
