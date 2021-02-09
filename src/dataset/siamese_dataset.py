import copy
import torch
from PIL import Image
from ..utils.logger import log


class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, args, df, transforms=None):
        self.args = args
        self.df = df
        self.transforms = transforms
        log("read dataset (%d records)" % len(self.df))
        log(f"dataset transforms: {self.transforms}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_paths = [row['image_1'], row['image_2']]
        ids = copy.copy(image_paths)
        image_dir = "%s/train_all" % self.args.data_dir
        image_paths = ["%s/%s/%s" % (image_dir, id.split('_')[0], id) for id in image_paths]
        # H * W * C
        images = [Image.open(path) for path in image_paths]
        # C * H * W
        images_ten = [self.transforms(image) for image in images]
        target = torch.tensor(row["label"], dtype=torch.float)
        return images_ten[0], images_ten[1], target, ids
