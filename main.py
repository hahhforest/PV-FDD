from ResnetSP import ResnetSP
from PVDataset import PVDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print("硬件：", device, "device")

# 数据集
dataset = PVDataset()
# 模型
model = ResnetSP(dataset.classes)
# 超参数
valid_r = 0.9
batch_size = 50
learning_rate = 0.001
num_epochs = 10

# 划分训练集以及测试集
train_size = int(len(dataset) * valid_r)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size, num_workers=0)
print('train_loader: ', len(train_loader.dataset))
print('test_loader: ', len(test_loader.dataset))

# 损失函数、优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

model.to(device)
loss_fn.to(device)

# 训练------------------------------------------
best_acc = 0.0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    running_loss = 0.0
    correct = 0.0

    for batch, (samples, labels) in enumerate(train_loader):
        samples = Variable(samples.to(device))
        labels = Variable(labels.to(device))
        # Backpropagation
        # 清空梯度信息，否则在每次进行反向传播时都会累加
        optimizer.zero_grad()
        # Compute prediction and loss
        output = model(samples)
        loss = loss_fn(output, labels)

        # loss反向传播
        loss.backward()
        # 梯度更新模型参数
        optimizer.step()

        # 状态信息
        running_loss += (loss.item() * labels.shape[0])
        prediction = output.argmax(1)
        correct += (prediction == labels).sum().item()

        if batch % 100 == 99:
            print(f'[Epoch {epoch+1}  Batch {batch+1}  Correct {correct}  Total {(batch+1) * batch_size}\
              Running_loss {loss.item()}]')
            # print('prediction: ', prediction)
            # print('labels: ', labels)
            # print('correct: ', (prediction == labels).sum().item())
    print(f"acc: {correct / len(train_loader.dataset)}  Loss: {running_loss / len(train_loader.dataset):>7f}")


# 测试------------------------------------------
num = len(test_loader.dataset)
test_loss, correct = 0.0, 0.0

with torch.no_grad():
    for samples, labels in test_loader:
        samples = Variable(samples.to(device))
        labels = Variable(labels.to(device))

        pred = model(samples)
        test_loss += (loss_fn(pred, labels).item()) * len(labels)
        correct += (pred.argmax(1) == labels).sum().item()

test_loss /= num
acc = correct / num
print("Test\n-------------------------------")
print(f"Accuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
