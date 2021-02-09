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
from .utils.util import save_model, load_model
from .utils.config import Config
from .models.transformer import BaseVit
from src import factory


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
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
    parser.add_argument('--train-all', type=bool, default=False)
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--droprate', type=float, default=0.5, help='drop rate')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use-gpu', type=bool, default=False)
    parser.add_argument('--model', type=str, default="TransNet")
    return parser.parse_args()









def train(args, model, train_dataloader, val_dataloader):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best = {
        'loss': float('inf'),
        'epoch': -1,
    }

    if args.resume_from:
        detail = load_model(args.resume_from, model, optim=optimizer)
        best.update({
            'loss': detail['loss'],
            'epoch': detail['epoch'],
            'accuracy': detail['accuracy'],
        })

    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(best['epoch']+1, args.epoch):
        log(f'\n----- epoch {epoch} -----')
        # set seed
        run_nn(args, 'train', model, train_dataloader, criterion, optimizer)
        with torch.no_grad():
            val = run_nn(args, 'valid', model, val_dataloader, criterion)

        detail = {
            'loss': val['loss'],
            'epoch': epoch,
            'accuracy': val['accuracy'],
        }
        if val['accuracy'] <= best['accuracy']:
            best.update(detail)
        save_model(model, optimizer, detail)
        log('[best] ep:%d loss:%.4f accuracy:%.4f' % (best['epoch'], best['loss'], best['accuracy']))
        scheduler.step()


def run_nn(cfg, mode, model, loader, criterion=None, optim=None, apex=None):
    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise

    t1 = time.time()
    losses = []
    ids_all = []
    targets_all = []
    outputs_all = []

    for i, (inputs, targets, ids) in enumerate(loader):
        if cfg.use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # for id1, id2, target in zip(ids[0], ids[1], targets):
        #     print(id1, id2, target)

        outputs = model(inputs)
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
            ids_all.extend(ids)

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i + 1) * (len(loader) - (i + 1)))
        progress = f'\r[{mode}] {i+1}/{len(loader)} {elapsed}(s) eta:{eta}(s) loss:{(np.sum(losses)/(i+1)):.6f} loss200:{(np.sum(losses[-200:])/(min(i+1,200))):.6f}\n'
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'ids': np.array(ids_all),
        'targets': np.array(targets_all),
        'outputs': np.array(outputs_all),
        'loss': np.sum(losses) / (i+1)
    }

    if mode in ['train', 'valid']:
        accuracy = np.sum(result['targets'] == np.round(result['outputs'])) / len(result['targets'])
        result.update({'accuracy': accuracy})

    return result


def main():
    args = get_args()
    cfg = Config.fromfile(args.config)
    # copy arguments to config
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.use_gpu = args.use_gpu
    cfg.gpu = args.gpu
    cfg.lr = args.lr
    cfg.epoch = args.epoch

    train_dataloader = factory.get_dataloader(cfg.data.train)
    valid_dataloader = factory.get_dataloader(cfg.data.valid)
    train_dataset = factory.get_dataset_df(cfg.data.train)
    num_classes = len(train_dataset['target'].unique())

    model = BaseVit(cfg.imgsize[0], cfg.patch_size, num_classes=num_classes)
    train(cfg, model, train_dataloader, valid_dataloader)




if __name__ == '__main__':
    print(sys.argv)
    main()