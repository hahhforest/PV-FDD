from typing import Tuple
import scipy.io as scio
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.utils.data.dataset import Dataset


class PVDataset(Dataset):
    def __init__(self) -> None:
        super(PVDataset, self).__init__()

        self.label_dict = {'sample_nor': 0,
                           'sample_open_circuit': 1,
                           'sample_short_circuit1': 2,
                           'sample_short_circuit2': 3,
                           'sample_Degradation1_String': 4,
                           'sample_Degradation2_Array': 5,
                           'sample_partial_shading1': 6,
                           'sample_partial_shading2': 7,
                           'sample_partial_shading3': 8}
        self.filename = 'datas/sample_label.mat'
        self.classes = len(list(self.label_dict.keys()))

        # 读取有序的数据集
        self.dataset = self.read_mat()
        self.datasize = self.__len__()
        # 打乱
        self.mess_up()

    def __getitem__(self, item) -> Tuple[Tensor, Tensor]:
        """根据索引返回一个样本及标签"""
        label = self.dataset[item, 0, 0, -1].to(torch.int64)
        # # 转化为单热向量
        # one_hot = self.label_to_onehot(label)
        return self.dataset[item, :, 0:-1], label

    def __len__(self) -> int:
        """返回数据集大小"""
        return self.dataset.shape[0]

    def label_to_onehot(self, label: int) -> Tensor:
        """返回标签对应的单热向量，类别从0开始编号"""
        label = int(label)
        target = torch.zeros(1, self.classes).int()
        index = torch.tensor([[label]])
        target.scatter_(1, index, 1)
        return target

    def read_mat(self) -> Tensor:
        """读取所有数据，返回Nx40x5张量，此时数据按照标签有序"""
        data = scio.loadmat(self.filename)
        keys = list(self.label_dict.keys())

        # 所有40x5样本拼接为Nx1x40x5的张量
        res_tensor = torch.randn(1, 1, 40, 5)
        # 遍历所有标签
        for key in keys:
            # # 测试用，减少数据加载时间
            # if key != 'sample_nor':
            #     break

            print("读取... " + key)
            row = data[key][0]
            # 遍历当前标签所有样本
            for sample in row:
                # 1x40x5
                sample = torch.unsqueeze(torch.from_numpy(sample), 0).type(torch.FloatTensor)
                # 1x1x40x5
                sample = torch.unsqueeze(sample, 0)
                res_tensor = torch.cat([res_tensor, sample], dim=0)
        res_tensor = res_tensor[1:]

        return res_tensor

    def mess_up(self) -> None:
        """打乱数据集"""
        idx = list(range(self.__len__()))
        np.random.shuffle(idx)
        self.dataset = self.dataset[idx]


if __name__ == '__main__':
    dataset = PVDataset()
    print(dataset.__getitem__(5))
