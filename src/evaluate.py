import os
import sys
import argparse
import copy
import time
import pickle

from tqdm import tqdm
import pandas as pd
import numpy as np
import PIL
from PIL import Image
import scipy.io as sio
import torch
from torchvision import transforms

from .models.attn_net import TransNet
from .utils.util import load_model
from .utils.logger import log, logger


"""
query image: image for querying
junk image: image in gallery that come from the same camera as the query image
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/Market-1501-v15.09.15")
    parser.add_argument("--use-gpu", type=bool, default=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--snapshot", type=str, default="checkpoints/attn_siamese_ep0.pt")
    parser.add_argument("--query-num", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="output")
    return parser.parse_args()


def display_image(path):
    img = Image.open(path)
    img.show()


def get_indices(query, gt_query_dir):
    """
    query format: 0001_c1s1_001051_00.jpg
    """
    # remove file extension
    # query = query.split('.')[0]
    good_path = "%s/%s_good.mat" % (gt_query_dir, query)
    junk_path = "%s/%s_junk.mat" % (gt_query_dir, query)

    good_indices = sio.loadmat(good_path)['good_index'].squeeze(0)
    junk_indices = sio.loadmat(junk_path)['junk_index'].squeeze(0)

    return good_indices, junk_indices


def compute_mAP(result, gt_query_dir):
    """
    1. remove junk indices from the query result
    2. match good shots using good indices
    3. calculate mAP

    result:
    {
        ids_1: [...],
        ids_2: [...],
        distances: [...],
        shot_ids: [...],
    }
    """
    dist_mat = {}
    for query, shot, dist, shot_id in tqdm(zip(
            result['ids_1'], result['ids_2'], result['distances'], result['shot_ids']), total=len(result['ids_1'])):
        query_name = query.split('.')[0]
        if query_name not in dist_mat:
            dist_mat[query_name] = np.full(shape=20000, fill_value=np.inf)
        dist_mat[query_name][shot_id] = dist

    ap_all = []
    for query in tqdm(result['ids_1']):
        query_name = query.split('.')[0]
        good_indices, junk_indices = get_indices(query_name, gt_query_dir)
        for index in junk_indices:
            dist_mat[query_name][index] = np.inf
        sorted_indices = np.argsort(dist_mat[query_name])

        # sorted_indices = np.array([True if (x in good_indices) else False for x in sorted_indices])
        mask = np.in1d(sorted_indices, good_indices)
        # print(sorted_indices)
        # print(good_indices)
        # print(mask)
        rows_good = np.argwhere(mask)

        ngood = len(good_indices)
        rows_good = rows_good.flatten()
        ap = 0
        for i in range(ngood):
            d_recall = 1.0 / ngood
            precision = (i + 1) * 1.0 / (rows_good[i] + 1)
            if rows_good[i] != 0:
                old_precision = i * 1.0 / rows_good[i]
            else:
                old_precision = 1.0
            ap = ap + d_recall * (old_precision + precision) / 2
        ap_all.append(ap)
    ap_all = np.array(ap_all)
    print(ap_all)
    print(np.mean(ap_all))


def get_transforms(args):
    test_transforms = [
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    return transforms.Compose(test_transforms)


class TestDataset(torch.utils.data.Dataset):
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
        image_paths = [
            "%s/query/%s" % (self.args.data_dir, row["query"]),
            "%s/bounding_box_test/%s" % (self.args.data_dir, row["shot"])
        ]
        # H * W * C
        images = [Image.open(path) for path in image_paths]
        # C * H * W
        images_ten = [self.transforms(image) for image in images]
        return images_ten[0], images_ten[1], [row["query"], row["shot"]], row['shot_id']


def get_model(args):
    model = TransNet(in_channels=3,
                     conv_channels=1024,
                     inner_channels=512,
                     fc_in_features=2048,
                     out_features=512,
                     backbone="resnet50")
    if args.snapshot:
        load_model(args.snapshot, model, args.use_gpu)
    return model


def get_dataset_df(args):
    query_dir = "%s/query" % args.data_dir
    gallery_dir = "%s/bounding_box_test" % args.data_dir
    df_test = {"query": [], "shot": [], "shot_id": []}
    gallery_dir_files = sorted(os.listdir(gallery_dir))
    for query in tqdm(os.listdir(query_dir)[:args.query_num]):
        for id, shot in enumerate(gallery_dir_files):
            if query.split('.')[-1] != 'jpg' or shot.split('.')[-1] != 'jpg':
                continue
            df_test["query"].append(query)
            df_test["shot"].append(shot)
            df_test["shot_id"].append(id)
    df_test = pd.DataFrame(df_test)
    df_test.to_csv("output/df_test.csv")
    return df_test


def run_nn(args, mode, model, loader, criterion=None, optim=None, apex=None):
    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise

    t1 = time.time()
    ids_1_all = []
    ids_2_all = []
    shot_ids_all = []
    dist_all = []

    for i, (inputs_1, inputs_2, images, shot_ids) in enumerate(loader):
        if args.use_gpu:
            inputs_1 = inputs_1.cuda()
            inputs_2 = inputs_2.cuda()

        batch_size = len(inputs_1)
        # NOTE: using cuda
        outputs = model(inputs_1, inputs_2)

        with torch.no_grad():
            dist_all.extend(outputs.cpu().numpy())
            ids_1_all.extend(images[0])
            ids_2_all.extend(images[1])
            shot_ids_all.extend(shot_ids)

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i + 1) * (len(loader) - (i + 1)))
        progress = f'\r[{mode}] {i+1}/{len(loader)} {elapsed}(s) eta:{eta}(s)\n'
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'ids_1': np.array(ids_1_all),
        'ids_2': np.array(ids_2_all),
        'distances': np.array(dist_all),
        'shot_ids': np.array(shot_ids_all),
    }

    return result


def main():
    args = get_args()
    data_dir = args.data_dir
    test_dir = "%s/bounding_box_test" % data_dir
    gt_query_dir = "%s/gt_query" % data_dir

    if args.use_gpu:
        torch.cuda.set_device(args.gpu)

    all_files = os.listdir(test_dir)
    all_files = sorted(all_files)
    # id = 6619
    # filename = all_files[id]
    # img_path = "%s/%s" % (test_dir, filename)
    # print(filename)
    # display_image(img_path)
    test_df = get_dataset_df(args)
    test_transforms = get_transforms(args)
    test_dataset = TestDataset(args, test_df, test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers)
    query, shot, images, ids = next(iter(test_dataloader))
    # print(query, shot, images, ids)
    model = get_model(args)
    # result = run_nn(args, 'test', model, test_dataloader)
    # with open("%s/preds_out.pkl" % args.outdir, "wb") as fb:
    #     pickle.dump(result, fb)
    with open("%s/preds_out.pkl" % args.outdir, "rb") as fb:
        result = pickle.load(fb)
    compute_mAP(result, gt_query_dir)


if __name__ == '__main__':
    print(sys.argv)
    main()
