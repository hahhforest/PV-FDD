import scipy.io as scio
from torch import Tensor
import torch
import torch.nn as nn
import torch.utils.data.dataset


def read_datas(path: str) -> Tensor:
    datafile = path
    data = scio.loadmat(datafile)
    data_keys = list(data.keys())[3:]

    for key in data_keys:
        row = data[key]
        print(row)
        # sample = data[0]
        # print(sample)
        # print(sample.shape)
        #
        # tensor = torch.from_numpy(sample)
        # # 添加channel维度
        # tensor = torch.unsqueeze(tensor, 0)


if __name__ == '__main__':
    read_datas('datas/sample_label.mat')