import os
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader
import scipy.io as sio
import torchvision.transforms as T
import src.dataset.custom_dataset


def get_dataset_df(cfg):
    if cfg.dataset_type == 'CustomDataset':
        dataset = {"target": [], "path": [], "id": []}
        dataset_path = cfg.imgdir
        with open(cfg.class_map, 'rb') as fb:
            class_map = pickle.load(fb)
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.split('.')[-1] != 'jpg':
                    continue
                target = file.split('_')[0]
                target = class_map[target]
                id = file.split('.')[0]
                # Note: limit data size when debug
                # if target > 10:
                #     continue
                path = "%s/%s" % (root, file)
                dataset["target"].append(target)
                dataset["id"].append(id)
                dataset["path"].append(path)
        return pd.DataFrame(dataset)

    if cfg.dataset_type == 'CustomTestDataset':
        dataset = {'path': [], 'id': [], 'is_query': [], 'index': []}
        query_dir = cfg.query_dir
        gallery_dir = cfg.gallery_dir
        for file in os.listdir(query_dir):
            if file.split('.')[-1] != 'jpg':
                continue
            path = "%s/%s" % (query_dir, file)
            id = file.split('.')[0]
            is_query = 1
            index = -1
            dataset['path'].append(path)
            dataset['id'].append(id)
            dataset['is_query'].append(is_query)
            dataset['index'].append(index)

        # remove non-pictures, then sort images in gallery
        gallery_files = os.listdir(gallery_dir)
        for file in gallery_files:
            if file.split('.')[-1] != 'jpg':
                gallery_files.remove(file)
        gallery_files = sorted(gallery_files)

        for index, file in enumerate(gallery_files):
            path = "%s/%s" % (gallery_dir, file)
            id = file.split('.')[0]
            is_query = 0
            dataset['path'].append(path)
            dataset['id'].append(id)
            dataset['is_query'].append(is_query)
            dataset['index'].append(index)
        return pd.DataFrame(dataset)


def get_gt_query(cfg):
    result = {}
    if cfg.from_mat:
        dir = cfg.gt_query_dir
        for file in os.listdir(dir):
            if file.split('.')[-1] != 'mat':
                continue
            filename_without_extension = file.split('.')[0]
            id = '_'.join(filename_without_extension.split('_')[:-1])
            type = filename_without_extension.split('_')[-1]
            if id not in result:
                result[id] = {}
            # Index is starting from 1 (^_^)
            if type == 'good':
                result[id][type] = sio.loadmat("%s/%s" % (dir, file))['good_index'].squeeze(0) - 1
            else:
                result[id][type] = sio.loadmat("%s/%s" % (dir, file))['junk_index'].squeeze(0) - 1
    else:
        query_dir = cfg.query_dir
        gallery = cfg.gallery
        gallery = sorted(os.listdir(gallery))

        gl = []
        gc = []
        for img in gallery:
            if img.split('.')[-1] != 'jpg':
                continue
            file_without_ext = img.split('.')[0]
            pid = file_without_ext.split('_')[0]
            camera_id = file_without_ext.split('_')[1][1]
            gl.append(pid)
            gc.append(camera_id)

        for query in os.listdir(query_dir):
            if query.split('.')[-1] != 'jpg':
                continue
            file_without_ext = query.split('.')[0]
            pid = file_without_ext.split('_')[0]
            query_camera_id = file_without_ext.split('_')[1][1]
            camera_index = np.argwhere(gc == query_camera_id)
            query_index = np.argwhere(gl == pid)
            junk_index1 = np.argwhere(gl == -1)
            junk_index2 = np.intersect1d(query_index, camera_index)
            junk_index = np.append(junk_index2, junk_index1)
            good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
            if file_without_ext not in result:
                result[file_without_ext] = {}
            result[file_without_ext]['good'] = good_index
            result[file_without_ext]['junk'] = junk_index

    return result


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(T, transform.name):
            return getattr(T, transform.name)
        else:
            return eval(transform.name)
    transforms = [get_object(transform)(**transform.params) for transform in cfg.transforms]
    return T.Compose(transforms)


def get_dataloader(cfg):
    if cfg.dataset_type in ['CustomDataset', 'CustomTestDataset']:
        dataset_df = get_dataset_df(cfg)
        dataset = getattr(src.dataset.custom_dataset, cfg.dataset_type)(cfg, dataset_df)
        # dataset = src.dataset.custom_dataset.CustomDataset(cfg, dataset_df)
        loader = DataLoader(dataset, **cfg.loader)
        return loader


if __name__ == '__main__':
    from .utils.config import  Config
    cfg = Config.fromfile('conf/transformer.py')
    # get_dataset_df(cfg.data.train)
    # loader = get_dataloader(cfg.data.test)
    #
    # print(cfg.data.test.loader.batch_size)
    # print(next(iter(loader)))

    get_gt_query(cfg)

