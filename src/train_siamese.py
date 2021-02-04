import sys
import argparse
import random
import time
import os
import copy

import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from .utils.logger import log, logger
from .models.ft_net import ft_net
from .models.attn_net import TransNet



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/Market-1501-v15.09.15')
    parser.add_argument('--data-dir', type=str, default='data/Market-1501-v15.09.15/pytorch')
    parser.add_argument('--pos-num', type=int, default=2)
    parser.add_argument('--neg-num', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--n-classes', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--droprate', type=float, default=0.5, help='drop rate')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use-gpu', type=bool, default=False)
    return parser.parse_args()


def get_dataset_df(args):
    data_dir = args.data_dir
    train_path = "%s/train_all" % data_dir
    pos_num, neg_num = args.pos_num, args.neg_num

    df_train = {"image_1": [], "image_2": [], "label": []}
    df_val = {"image_1": [], "image_2": [], "label": []}

    for label in os.listdir(train_path)[:args.n_classes]:
        image_dir = "%s/%s" % (train_path, label)
        image_list = os.listdir(image_dir)
        for image in image_list:
            image_path = "%s/%s" % (image_dir, image)
            # For train set
            # randomly sample pos_num positive samples
            image_pairs = random.choices(image_list, k=pos_num)
            for image_pair in image_pairs:
                df_train["image_1"].append(image)
                df_train["image_2"].append(image_pair)
                df_train["label"].append(1)
            # randomly sample neg_num negative samples
            sub_image_dirs = [r for r in os.listdir(train_path) if r != label]
            sub_image_dirs = random.choices(sub_image_dirs, k=neg_num)
            for dir in sub_image_dirs:
                image_pair = random.choice(os.listdir("%s/%s" % (train_path, dir)))
                df_train["image_1"].append(image)
                df_train["image_2"].append(image_pair)
                df_train["label"].append(0)
            # For valid set
            # randomly sample 1 positive samples
            image_pair = random.choice(image_list)
            df_val["image_1"].append(image)
            df_val["image_2"].append(image_pair)
            df_val["label"].append(1)
            # randomly sample 1 negative samples
            sub_image_dirs = [r for r in os.listdir(train_path) if r != label]
            sub_image_dir = random.choice(sub_image_dirs)
            image_pair = random.choice(os.listdir("%s/%s" % (train_path, sub_image_dir)))
            df_val["image_1"].append(image)
            df_val["image_2"].append(image_pair)
            df_val["label"].append(0)
    return pd.DataFrame(df_train), pd.DataFrame(df_val)


def get_transforms(args):
    train_transforms = [
        transforms.Resize((256, 128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    val_transforms = [
        transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


def save_model(model, optim, detail):
    path = "checkpoints/attn_siamese_ep%d.pt" % detail['epoch']
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "detail": detail,
    }, path)
    log("save model to %s" % path)


def load_model(path, model, optim=None):
    # remap everthing onto CPU
    state = torch.load(str(path), map_location=lambda storage, location: storage)
    model.load_state_dict(state["model"])
    if optim:
        log("loading optim")
        optim.load_state_dict(state["model"])
    else:
        log("not loading optim")
    model.cuda()
    detail = state["detail"]
    log("loaded model from %s" % path)
    return detail


class CustomDataset(torch.utils.data.Dataset):
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


def train(args, model, train_dataset, val_dataset, train_dataloader, val_dataloader):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best = {
        'loss': float('inf'),
        'accuracy': 0.0,
        'epoch': -1,
    }

    if args.resume_from:
        detail = load_model(args.resume_from, model, optim=optimizer)
        best.update({
            'loss': detail['loss'],
            'accuracy': detail['accuracy'],
            'epoch': detail['epoch']
        })

    log('train data: loaded %d records' % len(train_dataset))
    log('valid data: loaded %d records' % len(val_dataset))

    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = torch.nn.BCELoss()

    for epoch in range(best['epoch']+1, args.epoch):
        log(f'\n----- epoch {epoch} -----')
        # set seed
        run_nn(args, 'train', model, train_dataloader, criterion, optimizer)
        with torch.no_grad():
            val = run_nn(args, 'valid', model, val_dataloader, criterion)

        detail = {
            'accuracy': val['acc'],
            'loss': val['loss'],
            'epoch': epoch,
        }
        if val['loss'] <= best['loss']:
            best.update(detail)
        save_model(model, optimizer, detail)
        log('[best] ep:%d loss:%.4f accuracy:%.4f' % (best['epoch'], best['loss'], best['accuracy']))
        scheduler.step()


def run_nn(args, mode, model, loader, criterion=None, optim=None, apex=None):
    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise

    t1 = time.time()
    losses = []
    ids_1_all = []
    ids_2_all = []
    targets_all = []
    outputs_all = []

    for i, (inputs_1, inputs_2, targets, ids) in enumerate(loader):
        batch_size = len(inputs_1)
        # NOTE: using cuda
        outputs = model(inputs_1, inputs_2)

        if mode in ['train', 'valid']:
            loss = criterion(outputs, targets)
            with torch.no_grad():
                losses.append(loss.item())

        if mode in ['train']:
            # NOTE: apex
            loss.backward()
            optim.step()
            optim.zero_grad()

        with torch.no_grad():
            targets_all.extend(targets.cpu().numpy())
            outputs_all.extend(outputs.cpu().numpy())
            ids_1_all.extend(ids[0])
            ids_2_all.extend(ids[1])

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i + 1) * (len(loader) - (i + 1)))
        progress = f'\r[{mode}] {i+1}/{len(loader)} {elapsed}(s) eta:{eta}(s) loss:{(np.sum(losses)/(i+1)):.6f} loss200:{(np.sum(losses[-200:])/(min(i+1,200))):.6f}\n'
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'ids_1': np.array(ids_1_all),
        'ids_2': np.array(ids_2_all),
        'targets': np.array(targets_all),
        'outputs': np.array(outputs_all),
        'loss': np.sum(losses) / (i+1)
    }

    if mode in ['train', 'valid']:
        acc = np.sum(result['targets'] == np.round(result['outputs'])) / len(result['targets'])
        result.update({'acc': acc})
        print(result['targets'])
        print(result['outputs'])
        log(f"[%s] accuracy: %.4f" % (mode, acc))

    return result


if __name__ == '__main__':
    print(sys.argv)
    logger.setup("./log", "siamese_debug")
    writer = SummaryWriter('log/train_siamese')
    args = get_args()

    if args.use_gpu:
        torch.cuda.set_device(args.gpu)

    df_train, df_val = get_dataset_df(args)
    # id = df_train.iloc[1]['image_1']
    # image_dir = "%s/train_all" % args.data_dir
    # path = "%s/%s/%s" % (image_dir, id.split('_')[0], id)
    # image = Image.open(path)
    train_transforms, val_transforms = get_transforms(args)
    # arr_image = train_transform(image)
    # print(arr_image.shape)

    # Get datasets
    train_dataset = CustomDataset(args, df_train, train_transforms)
    val_dataset = CustomDataset(args, df_val, val_transforms)

    # Get dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 shuffle=True)

    input_1, input_2, targets, ids = next(iter(train_dataloader))

    # ids: list(tuple(input_1_ids), tuple(input_2_ids))
    print(input_1.shape, input_2.shape, targets.shape)

    model = TransNet(in_channels=3,
                     conv_channels=1024,
                     inner_channels=512,
                     fc_in_features=2048,
                     out_features=512,
                     backbone="resnet50")
    if args.use_gpu:
        model.cuda()
    # outputs = model(input_1, input_2)
    # writer.add_graph(model, (input_1, input_2))
    # writer.close()
    # criterion = torch.nn.BCELoss()
    # loss = criterion(outputs, targets)
    # print(loss.item())
    # inputs_1_all = []
    # inputs_1_all.extend(ids[0])
    # targets_all = []
    # outputs_all = []
    # with torch.no_grad():
    #     targets_all.extend(targets.cpu().numpy())
    #     outputs_all.extend(outputs.cpu().numpy())
    #
    # result = {
    #     'targets': np.array(targets_all),
    #     'outputs': np.array(outputs_all),
    # }

    # print(result['targets'])
    # print(np.round(result['outputs']))
    # print(np.sum(result['targets'] == np.round(result['outputs'])) / len(result['targets']))

    train(args, model, train_dataset, val_dataset, train_dataloader, val_dataloader)