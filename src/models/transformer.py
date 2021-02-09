import torch
from torchvision import models
from torch import nn
from vit_pytorch import ViT
from ..utils.logger import log


class BaseVit(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1):
        super(BaseVit, self).__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )

    def forward(self, input, mask=None):
        if mask:
            preds = self.vit(input, mask=mask)
        else:
            preds = self.vit(input)
        return preds


if __name__ == '__main__':
    img = torch.randn(1, 3, 256, 256)

    num_classes = 751

    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=num_classes,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    preds = v(img)
    print(preds)
    print(preds.shape)

    from src.utils.config import Config
    from src.factory import get_dataloader, get_dataset_df

    cfg = Config.fromfile('conf/transformer.py')
    loader = get_dataloader(cfg.data.train)
    inputs, targets, ids = next(iter(loader))

    train_dataset = get_dataset_df(cfg.data.train)
    num_classes = len(train_dataset['target'].unique())
    print("number of classes: %d" % num_classes)

    model = BaseVit(cfg.imgsize[0], cfg.patch_size, num_classes=num_classes)
    model.train(True)
    outputs = model(inputs)
    criterion = torch.nn.CrossEntropyLoss()
    print(criterion(outputs, targets))
