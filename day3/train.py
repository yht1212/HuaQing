import time
import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import ImageTxtDataset  # 保持不变

# === 1. 准备数据集 ===
train_data = ImageTxtDataset(
    r"G:\py\pythonProjectdemo\day3\dataset\train.txt",
    r"G:\py\pythonProjectdemo\day3\dataset\Images\train",
    transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
)

test_data = ImageTxtDataset(
    r"G:\py\pythonProjectdemo\day3\dataset\val.txt",
    r"G:\py\pythonProjectdemo\day3\dataset\val.txt",
    transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
)

# === 2. 数据集长度和类别数 ===
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度: {train_data_size}")
print(f"测试数据集的长度: {test_data_size}")

# 自动检测类别数
num_classes = max(train_data.labels) + 1
print(f"自动检测到的类别数: {num_classes}")

# === 3. 加载数据集 ===
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# === 4. 定义AlexNet模型 ===
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # ↓ [64, 128, 128]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # ↓ [64, 64, 64]
            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # ↓ [192, 64, 64]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # ↓ [192, 32, 32]
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # ↓ [384, 32, 32]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # ↓ [256, 32, 32]
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # ↓ [256, 32, 32]
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))  # 固定输出 [256, 2, 2]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),  # = 1024
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# === 5. 设备配置和模型初始化 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chen = AlexNet(num_classes=num_classes).to(device)
print("使用设备:", device)

# === 6. 损失函数和优化器 ===
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optim = torch.optim.SGD(chen.parameters(), lr=learning_rate, momentum=0.9)

# === 7. 训练参数和日志 ===
total_train_step = 0
total_test_step = 0
epoch = 10
writer = SummaryWriter("logs_train")

start_time = time.time()

# === 8. 训练循环 ===
for i in range(epoch):
    print(f"\n----- 第{i+1}轮训练开始 -----")
    chen.train()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = chen(imgs)
        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}步的训练loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    end_time = time.time()
    print(f"本轮训练时间: {end_time - start_time:.2f}秒")

    # === 9. 测试阶段 ===
    chen.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = chen(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_accuracy / test_data_size

    print(f"测试集平均loss: {avg_test_loss:.4f}")
    print(f"测试集正确率: {test_accuracy:.4f}")

    writer.add_scalar("test_loss", avg_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
    total_test_step += 1

    # === 10. 保存模型 ===
    torch.save(chen.state_dict(), "C:Users/Lenovo/Desktop/save.pth")
    print(f"模型已保存")

writer.close()
print("训练完成！")
