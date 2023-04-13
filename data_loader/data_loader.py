from .cifar10 import CIFAR10


def dataloader(dataset_name, model_config, env_config):
    train_dataset = None
    test_dataset = None
    if dataset_name == 'cifar10':
        db = CIFAR10('./data/', 
                     model_config=model_config, 
                     env_config=env_config)
        x_tr, y_tr, x_te, y_te = db.get_dataset()
    else:
        raise NotImplementedError()

    train_dataset = CustomDataset(x_tr, y_tr)
    test_dataset = CustomDataset(x_te, y_te)
    return train_dataset, test_dataset


from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'data': self.x[idx],
                'label': self.y[idx]}