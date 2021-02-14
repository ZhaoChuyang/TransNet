# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm
import time
import os
import scipy.io
import yaml
import math
from src.models.baseline import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test

# fp16
try:
    from apex.fp16_utils import *
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='data/Market-1501-v15.09.15/pytorch', type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--fp16', action='store_true', help='use fp16.')
parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()
###load config###
# load the training config
# config_path = os.path.join('./model', opt.name, 'opts.yaml')
# with open(config_path, 'r') as stream:
#     config = yaml.load(stream)
# opt.fp16 = config['fp16']
# opt.PCB = config['PCB']
# opt.use_dense = config['use_dense']
# opt.use_NAS = config['use_NAS']
# opt.stride = config['stride']
opt.fp16 = False
opt.PCB = False
opt.use_dense = False
opt.use_NAS = False
opt.stride = 2
opt.nclasses = 751

str_ids = opt.gpu_ids.split(',')
# which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

print('We use the scale: %s' % opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ############### Ten Crop
    # transforms.TenCrop(224),
    # transforms.Lambda(lambda crops: torch.stack(
    #   [transforms.ToTensor()(crop)
    #      for crop in crops]
    # )),
    # transforms.Lambda(lambda crops: torch.stack(
    #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
    #       for crop in crops]
    # ))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384, 192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                      ['gallery', 'query', 'multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16) for x in
                   ['gallery', 'query', 'multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16) for x in ['gallery', 'query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()


######################################################################
# Load model
# ---------------------------
def load_network(network):
    save_path = os.path.join('./checkpoints', name, 'net_%s.pth' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n, 512).zero_().cuda()
        if opt.PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_().cuda()  # we have six parts

        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic',
                                                          align_corners=False)
                outputs = model(input_img)
                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam, mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses)
else:
    model_structure = ft_net(opt.nclasses, stride=opt.stride)

if opt.PCB:
    model_structure = PCB(opt.nclasses)

# if opt.fp16:
#    model_structure = network_to_half(model_structure)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if opt.PCB:
    # if opt.fp16:
    #    model = PCB_test(model[1])
    # else:
    model = PCB_test(model)
else:
    # if opt.fp16:
    # model[1].model.fc = nn.Sequential()
    # model[1].classifier = nn.Sequential()
    # else:
    model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model, dataloaders['gallery'])
    query_feature = extract_feature(model, dataloaders['query'])
    if opt.multi:
        mquery_feature = extract_feature(model, dataloaders['multi-query'])

# Save to Matlab for check
result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
          'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
scipy.io.savemat('pytorch_result.mat', result)

print(opt.name)
# result = './model/%s/result.txt' % opt.name
# os.system('python evaluate_gpu.py | tee -a %s' % result)

query_dir = './data/Market-1501-v15.09.15/query'
gallery = './data/Market-1501-v15.09.15/bounding_box_test'
gallery = sorted(os.listdir(gallery))

gl = []
gc = []
gt_query = {}
for img in gallery:
    if img.split('.')[-1] != 'jpg':
        continue
    file_without_ext = img.split('.')[0]
    pid = file_without_ext.split('_')[0]
    camera_id = file_without_ext.split('_')[1][1]
    gl.append(pid)
    gc.append(camera_id)

gl = np.array(gl)
gc = np.array(gc)
for query in os.listdir(query_dir):
    if query.split('.')[-1] != 'jpg':
        continue
    file_without_ext = query.split('.')[0]
    pid = file_without_ext.split('_')[0]
    query_camera_id = file_without_ext.split('_')[1][1]
    camera_index = np.argwhere(gc == query_camera_id)
    query_index = np.argwhere(gl == pid)
    junk_index1 = np.argwhere(gl == '-1')
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    id = int(file_without_ext.split('_')[0])
    if id not in gt_query:
        gt_query[id] = {}
    gt_query[id]['good'] = good_index
    gt_query[id]['junk'] = junk_index

ap_all = []
gallery_size = len(result['gallery_f'])
for query_feature, query_label, query_cam in tqdm(zip(result['query_f'], result['query_label'], result['query_cam']), total=len(result['query_f'])):
    dist_vec = [0] * gallery_size
    for idx, (gallery_feature, gallery_label, gallery_cam) in tqdm(enumerate(zip(result['gallery_f'], result['gallery_label'], result['gallery_cam'])), total=len(result['gallery_f'])):
        dist_vec[idx] = gallery_feature
    print(query_label)
    dist_vec = np.array(dist_vec)
    dist_vec = np.linalg.norm(dist_vec - query_feature, axis=1)
    good_index = gt_query[query_label]['good']
    junk_index = gt_query[query_label]['junk']
    sorted_indices = np.argsort(dist_vec)
    mask = np.in1d(sorted_indices, junk_index, invert=True)
    sorted_indices = sorted_indices[mask]

    mask = np.in1d(sorted_indices, good_index)
    rows_good = np.argwhere(mask == True)
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

    ap_all.append(ap)

print('mAP: %.3f' % np.mean(ap_all))


if opt.multi:
    result = {'mquery_f': mquery_feature.numpy(), 'mquery_label': mquery_label, 'mquery_cam': mquery_cam}
    scipy.io.savemat('multi_query.mat', result)
