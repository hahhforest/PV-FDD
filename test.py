import torch
import torch.nn as nn
import scipy.io as scio
import numpy as np
from ResnetSP import ResnetSP
from PVDataset import PVDataset
from torch.utils.data import DataLoader


def test1():
    # A = torch.tensor([[[1,2,3,4,5,6]]])
    A = torch.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[13,14,15,16],[17,18,19,20],[21,22,23,24]]])
    B = torch.flatten(A, 0)
    C = torch.flatten(A, 1)

    print(A.shape)
    print(B)
    print(B.shape)
    print(C)
    print(C.shape)


def test2():
    conv1 = nn.Conv1d(in_channels=256, out_channels=100, kernel_size=(3,), stride=(1,), padding=0)
    input = torch.randn(32, 35, 256)  # [batch_size, max_len, embedding_dim]
    input = input.permute(0, 2, 1)  # 交换维度：[batch_size, embedding_dim, max_len]
    out = conv1(input)  # [batch_size, out_channels, n+2p-f/s+1]
    print(out.shape)  # torch.Size([32, 100, 33])


def test3():
    data = scio.loadmat('datas/sample_label.mat')
    label_dict = {'sample_nor': 0,
                  'sample_open_circuit': 1,
                  'sample_short_circuit1': 2,
                  'sample_short_circuit2': 3,
                  'sample_Degradation1_String': 4,
                  'sample_Degradation2_Array': 5,
                  'sample_partial_shading1': 6,
                  'sample_partial_shading2': 7,
                  'sample_partial_shading3': 8}
    keys = list(label_dict.keys())
    for key in keys:
        row = data[key][0]
        print(row[0])


def test4():
    a = torch.randn(40, 4)
    a = torch.unsqueeze(a, dim=0)
    c = torch.randn(1, 40, 4)
    d = torch.cat([a, c], dim=0)
    print(a)
    print(c)
    print(d)
    b = a[1:]
    print(b.shape)
    print(b)


def test5():
    a = torch.randn(5, 1, 1)
    idx = list(range(5))
    np.random.shuffle(idx)
    b = a[idx]

    print(a)
    print(b)


def test6():
    # self[index[i][j]][j] = src[i][j]  # if dim == 0
    # self[i][index[i][j]] = src[i][j] # if dim == 1
    src = torch.rand(2, 5)
    self = torch.zeros(3, 5)
    idx = torch.tensor([[0, 1, 2, 0, 0],
                        [2, 0, 0, 1, 2]])

    print(src)
    print(self.scatter_(0, idx, src))


def test7():
    targets = torch.zeros(1, 9).int()
    index = torch.tensor([[6]])
    targets.scatter_(1, index, 1)
    print(targets.size(), index.size())
    print(targets)


def test8():

    idx = 0
    idx = idx.astype('int64')
    print(type(idx))
    data = torch.randn(5, 40, 5)
    print(data[idx, :, 0:-1].shape)
    print(data)
    print(data[idx, 0, -1])


def test9():
    # 数据集
    dataset = PVDataset()
    # 模型
    model = ResnetSP(dataset.classes)
    # 超参数
    valid_r = 0.9
    batch_size = 100
    learning_rate = 0.001
    epochs = 5

    # 划分训练集以及测试集
    train_size = int(len(dataset) * valid_r)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=4)

    # 损失函数、优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)


def test10():
    # Example of target with class indices
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()
    print('input:\n', input)
    print('target:\n', target)
    print('output:\n', output)
    # Example of target with class probabilities
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5).softmax(dim=1)
    output = loss(input, target)
    output.backward()
    print('input:\n', input)
    print('target:\n', target)
    print('output:\n', output)


def test11():
    model = ResnetSP(num_classes=9)
    input = torch.randn(10, 1, 40, 4, requires_grad=True)
    out = model(input)

    print(out.shape)
    print(out)

    labels = torch.tensor([4, 1, 1, 1, 5, 5, 1, 1, 1, 4])

    prediction = torch.max(out, 1)[1]
    print(prediction)
    pre = out.argmax(1)
    print(pre)

    train_correct = (prediction == labels).sum()
    train_correct2 = (prediction == labels).sum().item()
    print(train_correct)
    print(train_correct2)

    print(len(labels))


def test12():
    dataset = PVDataset()
    dataloader = DataLoader(dataset, 10, num_workers=4)
    for i, (samples, labels) in enumerate(dataloader):
        print(f"i: {i}  batchsize: {len(labels)}")


def test13():
    output = torch.tensor([[-1.6361e+00, -2.1608e+00, -9.3933e-01, -1.3149e+00, -2.0593e+00,
          7.3286e+00,  4.0393e-01, -6.5758e+00, -1.0599e+01],
        [ 1.8431e+00, -1.5359e+00, -3.7258e-01, -8.3164e+00,  7.8449e+00,
         -7.8651e-01,  7.0739e-01, -6.5232e+00, -1.1977e+01],
        [ 1.8809e+00,  3.4583e+00, -1.5943e+01, -1.3582e+01, -5.4176e+00,
         -8.5732e+00, -6.9426e+00,  3.9310e+00,  1.4895e+01],
        [-4.2310e+00, -8.0593e+00, -1.0780e+01, -6.9946e+00, -3.2934e+00,
          1.8433e+00,  1.3671e+01,  5.4770e+00, -1.3587e+01],
        [-4.2433e+00, -7.9248e+00, -1.0983e+01, -7.0003e+00, -3.1982e+00,
          1.7871e+00,  1.3520e+01,  5.5909e+00, -1.3168e+01],
        [-2.4810e+00, -4.8366e+00, -1.9036e+01, -1.2344e+01, -1.1464e+01,
         -7.4502e+00,  6.3077e+00,  1.5916e+01,  2.7861e+00],
        [-1.6583e-01, -3.0753e+00,  5.9239e+00, -2.2937e-01,  3.7118e-01,
          2.8742e-01, -3.5701e+00, -1.2265e+01, -1.5289e+01],
        [ 9.0682e-02,  4.4894e+00, -1.0617e+01, -1.3193e+01, -2.0213e+00,
         -1.5290e+00, -8.8428e+00,  6.4356e-01,  1.1325e+01],
        [-1.0107e+00, -1.4275e+00,  5.3811e+00,  7.4507e-01,  6.9536e-01,
         -1.7289e+00, -2.5997e+00, -9.0744e+00, -1.1264e+01],
        [-6.5050e-01,  4.1703e+00, -1.0863e+01, -1.2228e+01, -2.3711e+00,
         -6.8263e-01, -7.8472e+00,  8.4326e-01,  1.1427e+01],
        [-1.3543e+00, -1.6192e+00, -1.1759e+00, -1.7779e+00, -1.9383e+00,
          6.7127e+00, -2.4949e-01, -6.3622e+00, -8.9982e+00],
        [-2.3399e-01, -1.7900e+00,  6.5904e+00,  2.1282e+00, -2.5021e+00,
         -2.9389e+00, -5.4881e+00, -9.5791e+00, -1.0027e+01],
        [ 1.6729e+00,  2.0805e+00, -1.5787e+01, -1.2957e+01, -5.6526e+00,
         -8.6535e+00, -5.6818e+00,  4.9072e+00,  1.4077e+01],
        [ 2.7383e+00,  3.6424e+00,  9.6742e-01, -8.0105e+00, -5.5569e-01,
         -2.6073e-01, -6.9438e+00, -6.2107e+00, -3.1791e+00],
        [ 7.0078e+00, -4.3116e-01, -3.7318e+00, -1.2066e+01,  4.3593e+00,
         -6.0350e+00, -5.7472e+00, -4.0085e+00, -4.3898e+00],
        [-3.6282e+00, -5.5214e+00, -8.0014e+00, -5.7148e+00, -3.0684e+00,
          5.1568e+00,  9.2038e+00,  1.7455e+00, -9.6777e+00],
        [-9.4030e+00, -9.3712e+00, -7.4856e+00,  3.1723e+00, -1.3331e+01,
          1.2417e+01,  4.9374e+00, -2.5128e-01, -1.4967e+01],
        [-8.5211e-01, -1.1445e+00,  5.4992e+00,  1.0262e+00,  1.2826e-01,
         -2.1302e+00, -3.1118e+00, -8.9591e+00, -1.0352e+01],
        [ 2.7764e+00, -2.1584e+00, -3.2915e+00, -7.9832e+00,  5.9375e+00,
         -2.8896e+00,  1.4414e+00, -3.5924e+00, -8.4069e+00],
        [-3.0865e+00, -5.2595e+00, -2.0086e+01, -1.1068e+01, -1.0666e+01,
         -6.4418e+00,  7.3771e+00,  1.5579e+01,  3.4421e+00],
        [ 4.6562e+00, -1.0358e-01, -1.1594e+00, -1.1807e+01,  6.2289e+00,
         -2.1349e+00, -3.7268e+00, -6.2926e+00, -8.5778e+00],
        [ 3.1493e+00,  3.1566e+00, -1.3490e+01, -1.5042e+01, -5.8352e+00,
         -1.0274e+01, -8.7832e+00,  4.4924e+00,  1.3905e+01],
        [ 2.8766e+00, -1.4610e+00, -2.7536e+00, -7.9588e+00,  5.4420e+00,
         -1.9135e+00, -1.5521e-01, -3.9446e+00, -7.4033e+00],
        [ 1.8426e-02,  6.0446e+00, -8.9812e-01, -6.4994e+00, -1.6652e+00,
         -4.4981e+00, -6.7659e+00, -4.3035e+00,  7.3483e-01],
        [ 9.6966e+00, -9.3945e-01, -3.8901e+00, -1.4152e+01,  4.4389e+00,
         -8.0108e+00, -8.4071e+00, -4.9383e+00, -5.3737e+00],
        [-9.3145e+00, -9.0143e+00, -7.4296e+00,  2.8776e+00, -1.3088e+01,
          1.2775e+01,  4.8132e+00, -2.2096e-01, -1.4931e+01],
        [ 9.1438e+00, -1.3014e+00, -4.9739e+00, -1.2778e+01,  4.0554e+00,
         -8.3752e+00, -6.2854e+00, -3.7412e+00, -4.5505e+00],
        [ 3.1776e+00,  2.6987e+00, -1.3530e+01, -1.5892e+01, -4.7850e+00,
         -9.9432e+00, -8.5968e+00,  4.5515e+00,  1.3069e+01],
        [-3.1441e+00, -7.8117e-01, -1.4142e+01, -1.2211e+01, -8.0482e+00,
         -1.7363e+00,  7.3310e-01,  1.0309e+01,  4.6947e+00],
        [-2.2322e+00,  5.9206e+00, -3.3659e+00, -5.1839e+00, -2.0256e+00,
         -2.8678e+00, -4.4867e+00, -3.4994e+00,  1.1025e+00],
        [-1.0546e+01, -5.6505e+00, -1.1802e+01,  2.4581e+00, -1.4023e+01,
          1.0995e+01,  1.7225e+00,  2.8229e+00, -3.8594e+00],
        [-6.6472e-01, -2.2772e+00,  6.4937e+00,  1.8370e+00, -1.1646e+00,
         -1.5626e+00, -4.4671e+00, -1.0383e+01, -1.1981e+01],
        [-6.2097e+00, -1.1107e+01,  7.6208e-02,  1.1288e+01, -1.2799e+01,
          2.2198e+00, -2.1962e+00, -9.2448e+00, -1.2695e+01],
        [-4.9026e+00, -6.2926e+00, -9.0051e+00, -3.5203e+00, -4.8508e+00,
          6.6589e+00,  9.5856e+00,  2.3429e+00, -1.0944e+01],
        [-3.1568e+00, -5.7501e+00, -7.0299e+00, -7.8602e+00,  1.2615e+00,
          1.7176e+00,  1.0862e+01,  9.3253e-01, -1.5117e+01],
        [ 1.3871e+00,  1.5703e+00, -1.6735e+00, -8.1850e+00,  6.5632e-01,
          3.7191e+00, -3.9425e+00, -6.1829e+00, -5.5539e+00],
        [ 1.4376e+00,  2.6019e+00,  5.0185e-01, -5.3765e+00,  3.5193e-02,
          7.7637e-01, -4.5059e+00, -5.6183e+00, -3.6807e+00],
        [ 1.0059e+01, -1.9996e+00, -4.2729e+00, -1.3140e+01,  4.6888e+00,
         -8.1049e+00, -7.0136e+00, -4.8919e+00, -6.0824e+00],
        [ 1.9536e+00,  4.4364e+00, -1.2279e+01, -1.6627e+01, -1.0963e+00,
         -4.5790e+00, -1.0284e+01,  1.1356e+00,  1.2162e+01],
        [-9.2265e-01, -3.5518e+00,  6.1604e+00,  1.1833e+00, -1.1713e+00,
          8.6524e-01, -4.1574e+00, -1.2412e+01, -1.5521e+01],
        [-2.1304e+00, -4.3624e+00, -1.8025e+01, -1.2388e+01, -1.1196e+01,
         -7.6598e+00,  5.7622e+00,  1.5430e+01,  2.3456e+00],
        [-2.8563e+00, -1.0223e+00, -1.3991e+01, -1.2523e+01, -8.1959e+00,
         -2.3166e+00,  8.0353e-01,  1.0524e+01,  3.9864e+00],
        [-2.3994e+00, -4.8073e+00, -1.8934e+01, -1.2980e+01, -1.1620e+01,
         -7.4499e+00,  5.9104e+00,  1.5995e+01,  2.6246e+00],
        [ 1.2994e-03, -1.3614e+00,  6.1310e+00,  1.3625e+00, -5.3181e+00,
         -1.5996e+00, -7.7056e+00, -1.0093e+01, -8.3288e+00],
        [ 2.9971e+00, -2.4062e+00, -9.4322e-01, -8.4731e+00,  7.4746e+00,
         -2.0697e+00,  7.5127e-01, -5.8915e+00, -1.1381e+01],
        [ 4.8909e+00,  2.2146e-01, -3.4016e+00, -1.2636e+01,  5.1582e+00,
         -6.0840e-01, -4.5219e+00, -5.4074e+00, -6.0304e+00],
        [-1.7778e+00, -2.0879e+00,  7.8214e-01, -3.0855e+00, -1.7341e+00,
          8.9675e+00, -3.7492e-01, -9.2077e+00, -1.5265e+01],
        [-6.0355e+00, -9.9600e+00, -1.7913e+00,  1.0281e+01, -1.4353e+01,
          2.3427e+00, -2.4664e+00, -6.7278e+00, -7.8965e+00],
        [-4.7510e+00, -8.9658e+00, -1.1495e+01, -7.0669e+00, -4.9042e+00,
          2.9952e+00,  1.4427e+01,  6.2119e+00, -1.4127e+01],
        [-1.0140e+01, -7.7704e+00, -1.0291e+01,  3.4456e+00, -1.4663e+01,
          9.9350e+00,  3.3589e+00,  2.6540e+00, -7.5223e+00]])
    labels = torch.tensor([5, 4, 8, 6, 6, 7, 2, 8, 2, 8, 5, 2, 8, 1, 0, 6, 5, 2, 4, 7, 4, 8, 4, 1,
        0, 5, 0, 8, 7, 1, 5, 2, 3, 6, 6, 4, 1, 0, 8, 2, 7, 7, 7, 2, 4, 0, 5, 3,
        6, 5])
    correct_loss = 0.1141
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, labels)
    predicts = (output.argmax(1))
    correct_num = (predicts == labels).sum().item()
    print(loss)
    print(predicts)
    print(correct_num)


if __name__ == '__main__':
    test13()
