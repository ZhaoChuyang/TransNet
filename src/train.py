import argparse
import torch
from torchvision import datasets, transforms
import time
import os
import matplotlib.pyplot as plt
from .models.ft_net import ft_net
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/Market-1501-v15.09.15')
    parser.add_argument('--data-dir', type=str, default='data/Market-1501-v15.09.15/pytorch')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--train-all', type=bool, default=False)
    parser.add_argument('--warm-epoch', type=int, default=0, help='the first K epoch that needs warm up')
    parser.add_argument('--droprate', type=float, default=0.5, help='drop rate')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')

    return parser.parse_args()


def main():
    args = get_args()
    # transforms
    transform_train_list = [
            #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
            transforms.Resize((256, 128), interpolation=3),
            transforms.Pad(10),
            transforms.RandomCrop((256,128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    transform_val_list = [
            transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    # if opt.PCB:
    #     transform_train_list = [
    #         transforms.Resize((384,192), interpolation=3),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]
    #     transform_val_list = [
    #         transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]

    # if opt.erasing_p>0:
    #     transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

    # if opt.color_jitter:
    #     transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

    # print(transform_train_list)

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    image_datasets = {}
    train_path = "%s/train_all" % args.data_dir if args.train_all else "%s/train" % args.data_dir
    val_path = "%s/val" % args.data_dir

    image_datasets['train'] = datasets.ImageFolder(train_path, data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(val_path, data_transforms['val'])

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True) # 8 workers may work faster
        for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    use_gpu = torch.cuda.is_available()

    since = time.time()

    inputs, classes = next(iter(dataloaders['train']))
    print(time.time() - since)
    print(f"input shape: {inputs.shape}")
    print(f"label size: {classes.shape}")

    model = ft_net(len(class_names), args.droprate, args.stride)

    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    model = train(args, image_datasets, dataloaders, model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=3)


def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = 'checkpoints/%s' % save_filename
    torch.save(network.cpu().state_dict(), save_path)


'''
DRAW CURVE
'''
y_loss = {'train': [], 'val': []}
y_err = {'train': [], 'val': []}
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig('checkpoints/%s.jpg' % current_epoch)


def train(args, datasets, dataloaders, model, criterion, optimizer, scheduler, num_epochs=3):
    since = time.time()
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    warm_up = 0.1
    warm_iteration = round(dataset_sizes['train'] / args.batch_size) * args.warm_epoch  # first 5 epoch

    for epoch in range(num_epochs):
        print("\n----- epoch %d -----" % epoch)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0
            running_corrects = 0

            for data in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                inputs, labels = data
                batch_size, c, h, w = inputs.shape
                # skip the last batch
                if batch_size < args.batch_size:
                    continue
                # print(f'shape of inputs: {inputs.shape}')
                # print(f'shape of labels: {labels.shape}')
                optimizer.zero_grad()

                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                # print(f'shape of outputs: {outputs.shape}')

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if epoch < args.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up
                if phase == 'train':
                    loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss.item() * batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print("%s loss: %.3f, acc: %.3f" % (phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1-epoch_acc)

            # save model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

            time_elapsed = time.time() - since

        time_elapsed = time.time() - since
        model.load_state_dict(last_model_wts)
        save_network(model, 'last')
        return model


if __name__ == '__main__':
    main()