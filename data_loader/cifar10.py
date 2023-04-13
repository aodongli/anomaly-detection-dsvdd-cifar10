from torchvision.datasets import CIFAR10 as CIFAR10Download
import torchvision.transforms as transforms
from PIL import Image
import os.path
import numpy as np
import torch
import torch.nn as nn
import kornia.augmentation as K


class CIFAR10:
    def __init__(self, root, model_config=None, env_config=None):
        self.model_config = model_config
        self.env_config = env_config

        data_transform = transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Lambda(lambda x: (x-0.5)*2) # match the diffusion input scale
                                            ])

        train_dataset = CIFAR10Download(download=False, root=root, train=True, transform=data_transform)
        test_dataset = CIFAR10Download(download=False, root=root, train=False, transform=data_transform)

        # split data by label
        def dataset2dict(dataset):
            res = {}
            for (x, y) in dataset:
                if y not in res:
                    res[y] = []
                res[y].append(x)
            return res
        # self.db_train = dataset2dict(train_dataset)
        # self.db_test = dataset2dict(test_dataset)

        def dataset2tensor(dataset):
            xs = [] # list of tensor
            ys = [] # list of int
            for (x,y) in dataset:
                xs.append(x)
                ys.append(y)
            return torch.stack(xs, 0), torch.tensor(ys)
        self.x_tr, self.y_tr = dataset2tensor(train_dataset)
        self.x_te, self.y_te = dataset2tensor(test_dataset)

        self.normal_cls = self.env_config.normal_cls if self.env_config is not None else 0
        self.contamination_ratio = self.env_config.contamination_ratio if self.env_config is not None else 0.1

    def get_dataset(self):
        x_tr_normal = self.x_tr[np.where(self.y_tr==self.normal_cls)]
        num_clean = x_tr_normal.shape[0]
        x_tr_abnormal = self.x_tr[np.where(self.y_tr!=self.normal_cls)]
        num_contamination = int(self.contamination_ratio/(1-self.contamination_ratio)*num_clean)
        idx_contamination = np.random.choice(np.arange(x_tr_abnormal.shape[0]), 
                                             num_contamination, 
                                             replace=False)
        x_tr_contamination = x_tr_abnormal[idx_contamination]
        x_tr = torch.cat([x_tr_normal, x_tr_contamination], 0)
        y_tr = torch.zeros(x_tr.shape[0])
        y_tr[num_clean:] = 1

        x_te = self.x_te
        y_te = torch.ones(x_te.shape[0])
        y_te[np.where(self.y_te==self.normal_cls)] = 0

        return [x_tr, y_tr, x_te, y_te]
