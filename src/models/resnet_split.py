import torch
from torch import nn
from torchvision import models
from copy import deepcopy


if __name__ == '__main__':
    resnet = models.resnet50(pretrained=True)
    print(resnet)
    input = torch.randn(size=(2, 3, 256, 128))
    input_1 = torch.randn(size=(2, 1024, 16, 8))
    resnet_conv = nn.Sequential(*list(resnet.children())[:-3])
    resnet_list = list(resnet.children())
    print(resnet_list)
    resnet_head = nn.Sequential(*list(resnet.children())[:-3])
    resnet_tail = nn.Sequential(*list(resnet.children())[-3:-1])

    def hook(module, input, output):
        print(f'input shape: {input[0].shape}')
        print(f'output shape: {output.shape}')
    # resnet.layer3.register_forward_hook(hook)
    # resnet.layer4.register_forward_hook(hook)
    # resnet(input)

    # inter_out = resnet_head(input)
    output = resnet_tail(input_1)
    print(f'inter_out shape: {output.shape}')
    exit(0)
    print(resnet_conv)

    input_1 = torch.randn(size=(2, 3, 256, 128))
    input_2 = torch.randn(size=(2, 3, 256, 128))