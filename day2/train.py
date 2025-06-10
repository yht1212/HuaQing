# 完整的模型训练套路(以CIFAR10为例)
import time

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from day2.alex import AlexNet
from model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../day2/dataset_chen",
                                         train=True,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

test_data = torchvision.datasets.CIFAR10(root="../day2/dataset_chen",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True )

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度{train_data_size}")
print(f"测试数据集的长度{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data,batch_size=64)
test_loader = DataLoader(test_data,batch_size=64)

# 创建网络模型

#chen = Chen()
chen = AlexNet(num_classes=10)


# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 1e-2 相当于(10)^(-2)
learning_rate = 0.01
optim = torch.optim.SGD(chen.parameters(),lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0 #记录训练的次数
total_test_step = 0 # 记录测试的次数
epoch = 10 # 训练的轮数

# 添加tensorboard
writer = SummaryWriter("../day2/logs_train")

# 添加开始时间
start_time = time.time()

for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")
    # 训练步骤
    for data in train_loader:
        imgs, targets = data
        outputs = chen(imgs)
        loss = loss_fn(outputs,targets)

        # 优化器优化模型
        optim.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}的训练的loss:{loss.item()}")
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    end_time = time.time()
    print(f"训练时间{end_time - start_time}")

    # 测试步骤（以测试数据上的正确率来评估模型）
    total_test_loss = 0.0
    # 整体正确个数
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            outputs = chen(imgs)
            # 损失
            loss = loss_fn(outputs,targets)
            total_test_loss += loss.item()
            # 正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy,total_test_step)
    total_test_step += 1

    # 保存每一轮训练模型
    torch.save(chen,"C:Users/Lenovo/Desktop/save.pth")
    print("模型已保存")

writer.close()

# 训练数据集的长度50000
# 测试数据集的长度10000
# -----第1轮训练开始-----
# 第500的训练的loss:2.3026061058044434
# 训练时间57.69292902946472
# 整体测试集上的loss:361.4917950630188
# 整体测试集上的正确率：0.10199999809265137
# 模型已保存
# -----第2轮训练开始-----
# 第1000的训练的loss:2.302380084991455
# 第1500的训练的loss:2.3039979934692383
# 训练时间119.05187058448792
# 整体测试集上的loss:361.4703154563904
# 整体测试集上的正确率：0.10119999945163727
# 模型已保存
# -----第3轮训练开始-----
# 第2000的训练的loss:2.304591417312622
# 训练时间181.30994367599487
# 整体测试集上的loss:361.45426845550537
# 整体测试集上的正确率：0.10409999638795853
# 模型已保存
# -----第4轮训练开始-----
# 第2500的训练的loss:2.3024327754974365
# 第3000的训练的loss:2.3019564151763916
# 训练时间244.15084671974182
# 整体测试集上的loss:361.41082644462585
# 整体测试集上的正确率：0.1128000020980835
# 模型已保存
# -----第5轮训练开始-----
# 第3500的训练的loss:2.3022143840789795
# 训练时间306.67578196525574
# 整体测试集上的loss:361.30562710762024
# 整体测试集上的正确率：0.12210000306367874
# 模型已保存
# -----第6轮训练开始-----
# 第4000的训练的loss:2.301095962524414
# 第4500的训练的loss:2.300522565841675
# 训练时间369.0325574874878
# 整体测试集上的loss:361.01451230049133
# 整体测试集上的正确率：0.12210000306367874
# 模型已保存
# -----第7轮训练开始-----
# 第5000的训练的loss:2.2971253395080566
# 训练时间431.5951557159424
# 整体测试集上的loss:358.6766059398651
# 整体测试集上的正确率：0.11630000174045563
# 模型已保存
# -----第8轮训练开始-----
# 第5500的训练的loss:2.284515380859375
# 第6000的训练的loss:2.219313383102417
# 训练时间501.2035791873932
# 整体测试集上的loss:333.31863617897034
# 整体测试集上的正确率：0.18479999899864197
# 模型已保存
# -----第9轮训练开始-----
# 第6500的训练的loss:2.1817586421966553
# 第7000的训练的loss:1.9778567552566528
# 训练时间595.3534014225006
# 整体测试集上的loss:324.90182173252106
# 整体测试集上的正确率：0.1987999975681305
# 模型已保存
# -----第10轮训练开始-----
# 第7500的训练的loss:1.9860937595367432
# 训练时间687.9043593406677
# 整体测试集上的loss:311.35134196281433
# 整体测试集上的正确率：0.22190000116825104
# 模型已保存