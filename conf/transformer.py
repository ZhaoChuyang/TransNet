net_type = 'single'
# resume_from = None
gt_query_dir = './data/Market-1501-v15.09.15/gt_query'
query_dir = './data/Market-1501-v15.09.15/query'
gallery = './data/Market-1501-v15.09.15/bounding_box_test'

imgsize = (256, 256)
patch_size = 32
batch_size = 16
num_workers = 0

resize = dict(name='Resize', params=dict(size=imgsize, interpolation=3))
pad = dict(name='Pad', params=dict(padding=10))
crop = dict(name='RandomCrop', params=dict(size=imgsize))
hflip = dict(name='RandomHorizontalFlip', params=dict())
totensor = dict(name='ToTensor', params=dict())
normalize = dict(name='Normalize', params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))


data = dict(
    train_all=dict(
        class_map='./cache/train_class_map.pkl',
        dataset_type='CustomDataset',
        imgdir='./data/Market-1501-v15.09.15/pytorch/train_all',
        imgsize=imgsize,
        transforms=[resize, pad, crop, hflip, totensor, normalize],
        loader=dict(
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=False,
        )
    ),
    train=dict(
        class_map='./cache/train_class_map.pkl',
        dataset_type='CustomDataset',
        imgdir='./data/Market-1501-v15.09.15/pytorch/train',
        imgsize=imgsize,
        transforms=[resize, pad, crop, hflip, totensor, normalize],
        loader=dict(
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=False,
        )
    ),
    valid=dict(
        class_map='./cache/train_class_map.pkl',
        dataset_type='CustomDataset',
        imgdir='./data/Market-1501-v15.09.15/pytorch/val',
        imgsize=imgsize,
        transforms=[resize, totensor, normalize],
        loader=dict(
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    ),
    test=dict(
        class_map='./cache/test_class_map.pkl',
        dataset_type='CustomTestDataset',
        query_dir='./data/Market-1501-v15.09.15/query',
        gallery_dir='./data/Market-1501-v15.09.15/bounding_box_test',
        imgsize=imgsize,
        transforms=[resize, totensor, normalize],
        loader=dict(
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    )
)
