import torch
from PIL import Image
from ..utils.logger import log
import src.factory


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.df = df
        self.transforms = src.factory.get_transforms(cfg)
        log("read dataset (%d records)" % len(self.df))
        log(f"dataset transforms: {self.transforms}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path'])
        img_ten = self.transforms(img)
        target = row['target']
        id = row['id']
        return img_ten, torch.tensor(target, dtype=torch.int64), id


class CustomTestDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.df = df
        self.transforms = src.factory.get_transforms(cfg)
        log("read dataset (%d records)" % len(self.df))
        log(f"dataset transforms: {self.transforms}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path'])
        img_ten = self.transforms(img)
        id = row['id']
        is_query = row['is_query']
        index = row['index']
        return img_ten, id, is_query, index