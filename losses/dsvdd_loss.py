import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterDistLoss(nn.Module):
    def __init__(self, model_config=None, env_config=None):
        super(CenterDistLoss, self).__init__()

    def forward(self, z, center):
        dist = torch.sum((z - center) ** 2, dim=1)
        return dist