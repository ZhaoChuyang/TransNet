import os
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as T
import src.dataset.custom_dataset


def get_dataset_df(cfg):
    if cfg.dataset_type == 'CustomDataset':
        dataset = {"target": [], "path": [], "id": []}
        dataset_path = cfg.imgdir
        class_map = {}
        cnt = 0
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.split('.')[-1] != 'jpg':
                    continue
                target = file.split('_')[0]
                if target not in class_map:
                    class_map[target] = cnt
                    cnt += 1
                target = class_map[target]
                id = file.split('.')[0]
                path = "%s/%s" % (root, file)
                dataset["target"].append(target)
                dataset["id"].append(id)
                dataset["path"].append(path)
        return pd.DataFrame(dataset)


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(T, transform.name):
            return getattr(T, transform.name)
        else:
            return eval(transform.name)
    transforms = [get_object(transform)(**transform.params) for transform in cfg.transforms]
    return T.Compose(transforms)


def get_dataloader(cfg):
    if cfg.dataset_type == 'CustomDataset':
        dataset_df = get_dataset_df(cfg)
        dataset = src.dataset.custom_dataset.CustomDataset(cfg, dataset_df)
        loader = DataLoader(dataset, **cfg.loader)
        return loader


if __name__ == '__main__':
    from .utils.config import  Config
    cfg = Config.fromfile('conf/transformer.py')
    get_dataset_df(cfg.data.train)
    loader = get_dataloader(cfg.data.train)

