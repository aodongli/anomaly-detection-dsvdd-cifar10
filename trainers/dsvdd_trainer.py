import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


class DSVDD_Trainer:
    def __init__(self, model_config, env_config):
        self.device = torch.device(model_config['device'])
        model_class = model_config['model_class']
        self.model = model_class(model_config, env_config).to(self.device)
        loss_class = model_config['loss_class']
        self.loss = loss_class(model_config, env_config).to(self.device)
        optim_class = model_config['optim_class']
        self.optimizer = optim_class(self.model.parameters(),
                                     lr=model_config['learning_rate'], 
                                     weight_decay=model_config['l2'])

        self.model_config = model_config
        self.env_config = env_config

        self.exp_path = model_config['exp_path']
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
        # else:
        #     print(
        #         f"File {json_results} already present! Shutting down to prevent loss of previous experiments")


    def train_one_epoch(self, train_dataloader):
        self.model.train()
        loss_list = []
        for minibatch in train_dataloader:
            data = minibatch['data'].to(self.device)
            label = minibatch['label'].float().to(self.device)
            z, center = self.model(data)
            dist = self.loss(z, center)
            loss = torch.cat([dist[label == 0], 1/dist[label == 1]], 0)

            loss = loss.mean()
            loss_list.append(loss.detach().cpu().item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return np.mean(loss_list)


    def train(self, train_dataset, val_dataset=None, test_dataset=None):

        self.model.train()

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.model_config['batch_size'], 
                                      shuffle=True,
                                      drop_last=False)

        for epoch in range(self.model_config['epoch']):
            print(f'Epoch {epoch}:', end=' | ')

            loss_epoch = self.train_one_epoch(train_dataloader)
            print(loss_epoch, end=' | ')

            if val_dataset is not None:
                auc = self.test(val_dataset)
                print('val auc:', auc, end=' | ')

            if test_dataset is not None:
                auc = self.test(test_dataset)
                print('test auc:', auc)

        torch.save(self.model.state_dict(), self.exp_path + '/model.pt')


    def test(self, test_dataset, ckpt_path=''):
        if ckpt_path != '':
            self.model.load_state_dict(torch.load(ckpt_path))

        self.model.eval()

        test_dataloader = DataLoader(test_dataset,
                                  batch_size=self.model_config['batch_size'], 
                                  shuffle=False,
                                  drop_last=False)

        scores, labels = [], []
        with torch.no_grad():
            for minibatch in test_dataloader:
                data = minibatch['data'].to(self.device)
                label = minibatch['label'].to(self.device)

                z, center = self.model(data)
                dist = self.loss(z, center)

                scores.append(dist.cpu().numpy())
                labels.append(label.cpu().numpy())

        auc = roc_auc_score(np.concatenate(labels), np.concatenate(scores))
        return auc
