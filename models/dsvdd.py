import torch
import torch.nn as nn
import torch.nn.functional as F


class DSVDD_LeNet(nn.Module):

    def __init__(self, model_config, env_config):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=True, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=True)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=True, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=True)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=True, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=True)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=True)

        self.center = nn.Parameter(torch.ones(self.rep_dim))
        nn.init.normal_(self.center)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x, self.center

