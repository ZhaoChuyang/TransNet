import torch
from torchvision import models
from torch import nn
from ..utils.logger import log


class TransNet(nn.Module):
    def __init__(self, in_channels, conv_channels, inner_channels, fc_in_features, out_features, backbone='resnet34', pretrained=True):
        super(TransNet, self).__init__()

        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.inner_channels = inner_channels
        self.fc_in_features = fc_in_features
        self.out_features = out_features

        log('backbone model: %s' % backbone)
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.conv_stem = nn.Sequential(*list(self.backbone.children())[:-3])
            self.cls_net = nn.Sequential(*list(self.backbone.children())[-3:-1])
        if backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.conv_stem = nn.Sequential(*list(self.backbone.children())[:-3])
            self.cls_net = nn.Sequential(*list(self.backbone.children())[-3:-1])

        self.W_VA = nn.Conv1d(in_channels=conv_channels, out_channels=inner_channels, kernel_size=1, stride=1, padding=0)
        self.W_VB = nn.Conv1d(in_channels=conv_channels, out_channels=inner_channels, kernel_size=1, stride=1, padding=0)
        self.W_K = nn.Conv1d(in_channels=conv_channels, out_channels=inner_channels, kernel_size=1, stride=1, padding=0)
        self.W_Q = nn.Conv1d(in_channels=conv_channels, out_channels=inner_channels, kernel_size=1, stride=1, padding=0)
        self.W_A = nn.Conv1d(in_channels=inner_channels, out_channels=conv_channels, kernel_size=1, stride=1, padding=0)
        self.W_B = nn.Conv1d(in_channels=inner_channels, out_channels=conv_channels, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(1)
        # in_features = 2048, out_features = 512
        self.fc = nn.Linear(in_features=fc_in_features, out_features=out_features, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_1, input_2):
        batch_size = input_1.shape[0]

        features_1 = self.conv_stem(input_1)
        features_2 = self.conv_stem(input_2)

        features_h = features_1.shape[2]
        features_w = features_1.shape[3]
        # (B, C, H, W) <-> (1, 1024, 16, 8)
        # log(f'image 1 feature map shape: {features_1.shape}')
        # log(f'image 2 feature map shape: {features_2.shape}')
        # (B, C, N) <-> (1, 1024, 128)
        features_1 = torch.flatten(features_1, 2, -1)
        features_2 = torch.flatten(features_2, 2, -1)
        # log(f'image 1 feature shape (flatten): {features_1.shape}')
        # log(f'image 2 feature shape (flatten): {features_2.shape}')
        # define W_f, W_g and W_h as 1x1 conv
        # NOTE: change conv1d to conv2d?
        # (B, C', N) <-> (1, 512, 128)
        VA = self.W_VA(features_1)
        VB = self.W_VB(features_2)
        K = self.W_K(features_1)
        Q = self.W_Q(features_2)
        # log(f'VA shape: {VA.shape}')
        # log(f'VB shape: {VB.shape}')
        # log(f'K shape: {K.shape}')
        # log(f'Q shape: {Q.shape}')
        # S : (B, N, N) <-> (1, 128, 128)
        # S = transpose(K) * Q <-> (C, N, N) = (B, N, C') * (B, C', N)
        # NOTE: S is first flattened, then recovered to original dimensions
        # S_{i,j}: correlation between K_i and Q_j
        S = torch.matmul(torch.transpose(K, 1, 2), Q)
        n_features = S.shape[1]
        S = self.softmax(S.flatten(1))
        # log(f"sum of S: {torch.sum(S)}")
        S = S.view(batch_size, n_features, n_features)
        # recover O to (B, C, N)
        # O can be seen as the attended version of B
        output_B = torch.matmul(VA, S)
        VB = torch.transpose(VB, 1, 2)
        output_A = torch.transpose(torch.matmul(S, VB), 1, 2)
        # apply 1x1 conv for increasing dimensions of Output_A and Output_B
        output_A = self.W_A(output_A)
        output_B = self.W_B(output_B)
        # print(f"shape of O_A: {output_A.shape}")
        # print(f"shape of O_B: {output_B.shape}")

        output_A = output_A.view(batch_size, -1, features_h, features_w)
        output_B = output_B.view(batch_size, -1, features_h, features_w)

        # print(f"shape of O_A: {output_A.shape}")
        # print(f"shape of O_B: {output_B.shape}")

        features_A = self.cls_net(output_A)
        features_B = self.cls_net(output_B)
        features_A = features_A.squeeze(-1)
        features_A = features_A.squeeze(-1)
        features_B = features_B.squeeze(-1)
        features_B = features_B.squeeze(-1)
        # print(f"A features shape: {features_A.shape}")
        # print(f"A features shape: {features_B.shape}")

        features_A = self.fc(features_A)
        features_B = self.fc(features_B)

        dist = torch.linalg.norm(features_A - features_B, dim=1)
        # output = self.sigmoid(dist)
        return dist


if __name__ == '__main__':
    resnet = models.resnet50(pretrained=True)
    print(resnet)
    resnet_conv = nn.Sequential(*list(resnet.children())[:-3])
    print(resnet_conv)

    input_1 = torch.randn(size=(2, 3, 256, 128))
    input_2 = torch.randn(size=(2, 3, 256, 128))

    batch_size = input_1.shape[0]

    features_1 = resnet_conv(input_1)
    features_2 = resnet_conv(input_2)
    # (B, C, H, W) <-> (1, 1024, 16, 8)
    print(f'image 1 feature map shape: {features_1.shape}')
    print(f'image 2 feature map shape: {features_2.shape}')

    features_h = features_1.shape[2]
    features_w = features_1.shape[3]

    # (B, C, N) <-> (1, 1024, 128)
    features_1 = torch.flatten(features_1, 2, -1)
    features_2 = torch.flatten(features_1, 2, -1)
    print(f'image 1 feature shape (flatten): {features_1.shape}')
    print(f'image 2 feature shape (flatten): {features_2.shape}')
    # define W_f, W_g and W_h as 1x1 conv
    # NOTE: change conv1d to conv2d?
    in_channels = features_1.shape[1]
    print(f"stem output channels: {features_1.shape[1]}")
    out_channels = 512
    W_VA = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    W_VB = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    W_K = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    W_Q = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    W_A = nn.Conv1d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
    W_B = nn.Conv1d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
    # (B, C', N) <-> (1, 512, 128)
    VA = W_VA(features_1)
    VB = W_VB(features_2)
    K = W_K(features_1)
    Q = W_Q(features_2)
    print(f'VA shape: {VA.shape}')
    print(f'VB shape: {VB.shape}')
    print(f'K shape: {K.shape}')
    print(f'Q shape: {Q.shape}')
    # S : (B, N, N) <-> (1, 128, 128)
    # S = transpose(K) * Q <-> (C, N, N) = (B, N, C') * (B, C', N)
    # NOTE: S is first flattened, then recovered to original dimensions
    # S_{i,j}: correlation between K_i and Q_j
    S = torch.matmul(torch.transpose(K, 1, 2), Q)
    n_features = S.shape[1]
    S = S.flatten(1)
    m = nn.Softmax(1)
    S = m(S)
    print(f"sum of S: {torch.sum(S)}")
    S = S.view(batch_size, n_features, n_features)
    # recover O to (B, C, N)
    # O can be seen as the attended version of B
    output_B = torch.matmul(VA, S)
    VB = torch.transpose(VB, 1, 2)
    output_A = torch.transpose(torch.matmul(S, VB), 1, 2)
    # apply 1x1 conv for increasing dimensions of Output_A and Output_B
    output_A = W_A(output_A)
    output_B = W_B(output_B)
    print(f"shape of O_A: {output_A.shape}")
    print(f"shape of O_B: {output_B.shape}")

    resnet_cls = nn.Sequential(*list(resnet.children())[-3:-1])
    print(resnet_cls)
    output_A = output_A.view(batch_size, in_channels, features_h, features_w)
    output_B = output_B.view(batch_size, in_channels, features_h, features_w)

    print(f"shape of O_A: {output_A.shape}")
    print(f"shape of O_B: {output_B.shape}")

    features_A = resnet_cls(output_A)
    features_B = resnet_cls(output_B)
    features_A = features_A.squeeze(-1)
    features_A = features_A.squeeze(-1)
    features_B = features_B.squeeze(-1)
    features_B = features_B.squeeze(-1)
    print(f"A features shape: {features_A.shape}")
    print(f"A features shape: {features_B.shape}")

    fc = nn.Linear(in_features=2048, out_features=512, bias=True)

    features_A = fc(features_A)
    features_B = fc(features_B)

    dist = torch.linalg.norm(features_A - features_B, dim=1)
    print(dist)