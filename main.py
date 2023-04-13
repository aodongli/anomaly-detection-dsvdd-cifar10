import argparse
import os
import numpy as np
from data_loader.data_loader import dataloader
from config.parser import model_config_reader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_cifar10.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='cifar10')
    parser.add_argument('--contamination-ratio', dest='contamination_ratio', type=float, default=0.1)
    parser.add_argument('--normal-cls', dest='normal_cls', type=int, default=1)
    parser.add_argument('--ckpt-path', dest='ckpt_path', default='')
    return parser.parse_args()


def run_dataset(train_dataset, test_dataset, env_config, model_config):
    trainer_class = model_config['trainer_class']
    trainer = trainer_class(model_config, env_config)
    if env_config.ckpt_path != '':
        res = trainer.test(test_dataset, ckpt_path=env_config.ckpt_path)
    else:
        trainer.train(train_dataset, test_dataset=test_dataset)
        res = trainer.test(test_dataset)
    print('result:', res)
    return res


if __name__ == "__main__":
    env_config = get_args()
    model_config = model_config_reader(env_config.config_file)

    train_dataset, test_dataset = dataloader(env_config.dataset_name, model_config, env_config)
    res = run_dataset(train_dataset, test_dataset, env_config, model_config)