import torch
from torch import nn


if __name__ == '__main__':

    # 输入: [时间, 股票个数, 指标]
    # 假设:
    # 1. 时间是整数 (torch.int)
    # 2. 股票个数是整数 (torch.int)
    # 3. 指标是浮点数 (torch.float)

    # LSTM的输入需要是矩阵形式, 以上三个特征需要融合到一个矩阵中, 可以考虑将三个特征水平拼接
    # 时间可以使用 one-hot 形式向量表示, 或者数值表示
    # 股票个数使用数值表示
    # 指标个数使用数值表示
    # 输入特征为三维向量? 输入特征过少

    exit(0)
