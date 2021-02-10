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
from torch import nn
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from .utils.logger import log, logger
from src.utils import util
from .utils.config import Config
from src.models.transformer import BaseVit
from src.models.ft_net import ft_net
from src import factory


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='data/Market-1501-v15.09.15')
    parser.add_argument('--data-dir', type=str, default='data/Market-1501-v15.09.15/pytorch')
    parser.add_argument('--pos-num', type=int, default=2)
    parser.add_argument('--neg-num', type=int, default=1)
    # parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--n-classes', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--train-all', type=bool, default=False)
    parser.add_argument('--snapshot', type=str)
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--droprate', type=float, default=0.5, help='drop rate')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use-gpu', type=bool, default=False)
    parser.add_argument('--model', type=str, default="BaseViT")
    return parser.parse_args()


def test(cfg, model):
    assert cfg.snapshot
    util.load_model(cfg.snapshot, model, cfg.use_gpu)
    if cfg.model == 'ft_net':
        # This should output 512 dimensional features.
        model.classifier.classifier = nn.Sequential()
    elif cfg.model == 'BaseViT':
        pass
    model.eval()

    loader_test = factory.get_dataloader(cfg.data.test)
    gt_query = factory.get_gt_query(cfg)

    query_outputs = []
    gallery_outputs = []
    query_ids = []
    gallery_ids = []
    gallery_indices = []

    for inputs, ids, is_query, indices in loader_test:
        outputs = model(inputs)
        with torch.no_grad():
            outputs = outputs.cpu().numpy()
        for output, id, label, index in zip(outputs, ids, is_query, indices):
            # query image
            if label == 1:
                query_outputs.append(output)
                query_ids.append(id)
            # gallery image
            else:
                gallery_outputs.append(output)
                gallery_ids.append(id)
                gallery_indices.append(index)

    gallery_size = len(gallery_outputs)
    for query_id, query_output in zip(query_ids, query_outputs):
        dist_vec = [0] * gallery_size
        for gallery_output, gallery_index in zip(gallery_outputs, gallery_indices):
            dist_vec[gallery_index] = gallery_output
        dist_vec = np.array(dist_vec)
        dist_vec = np.linalg.norm(dist_vec - query_output, axis=1)
        good_index = gt_query[query_id]['good']
        junk_index = gt_query[query_id]['junk']

        dist_vec[junk_index] = np.inf
        sorted_indices = np.argsort(dist_vec)
        mask = np.in1d(sorted_indices, good_index)
        rows_good = np.argwhere(mask==True)
        rows_good = rows_good.flatten()

        ngood = len(good_index)
        ap = 0

        for i in range(ngood):
            d_recall = 1.0 / ngood
            precision = (i + 1) * 1.0 / (rows_good[i] + 1)
            if rows_good[i] != 0:
                old_precision = i * 1.0 / rows_good[i]
            else:
                old_precision = 1.0
            ap = ap + d_recall * (old_precision + precision) / 2

        print(ap)


def train(args, model, train_dataloader, val_dataloader):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best = {
        'loss': float('inf'),
        'epoch': -1,
        'accuracy': 0,
    }

    if args.resume_from:
        detail = util.load_model(args.resume_from, model, optim=optimizer)
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
        if val['accuracy'] >= best['accuracy']:
            best.update(detail)
        util.save_model(model, optimizer, args.model, detail)
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
            # NOTE: use apex here
            loss.backward()
            optim.step()
            optim.zero_grad()

        with torch.no_grad():
            targets_all.extend(targets.cpu().numpy())
            outputs = np.argmax(outputs.cpu().numpy(), axis=1)
            outputs_all.extend(outputs)
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
        accuracy = np.sum(result['targets'] == result['outputs']) / len(result['targets'])
        result.update({'accuracy': accuracy})

    return result


def main():
    args = get_args()
    cfg = Config.fromfile(args.config)

    # copy arguments to config
    cfg.mode = args.mode
    cfg.use_gpu = args.use_gpu
    cfg.gpu = args.gpu
    cfg.lr = args.lr
    cfg.epoch = args.epoch
    cfg.model = args.model
    if args.snapshot:
        cfg.snapshot = args.snapshot

    if cfg.mode == 'train':
        train_dataloader = factory.get_dataloader(cfg.data.train)
        valid_dataloader = factory.get_dataloader(cfg.data.valid)
        train_dataset = factory.get_dataset_df(cfg.data.train)
        num_classes = len(train_dataset['target'].unique())

        if cfg.model == 'BaseViT':
            model = BaseVit(cfg.imgsize[0], cfg.patch_size, num_classes=num_classes)
        elif cfg.model == 'ft_net':
            model = ft_net(num_classes)

        if cfg.use_gpu:
            torch.cuda.set_device(cfg.gpu)
            model.cuda()

        train(cfg, model, train_dataloader, valid_dataloader)

    if cfg.mode == 'test':
        loader_test = factory.get_dataloader(cfg.data.test)
        # num_classes takes no effect in test mode. But you should ensure
        # this value is equal to the value you set in the training stage,
        # otherwise errors will raise when loading saved model weights.
        num_classes = 751

        if cfg.model == 'BaseViT':
            model = BaseVit(cfg.imgsize[0], cfg.patch_size, num_classes=num_classes)
        elif cfg.model == 'ft_net':
            model = ft_net(num_classes)

        if cfg.use_gpu:
            torch.cuda.set_device(cfg.gpu)
            model.cuda()

        test(cfg, model)


if __name__ == '__main__':
    print(sys.argv)
    main()