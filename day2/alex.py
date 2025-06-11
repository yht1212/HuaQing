import torch
from torch import nn

class alex(nn.Module):
    def __init__(self, num_classes=10):
        super(alex, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 输出 32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 输出 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 输出 16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 输出 8x8

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 输出 8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),         # 输出 4x4

            nn.Flatten(),                        # 展平为 256 * 4 * 4 = 4096
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    alexnet = alex(num_classes=10)
    y = alexnet(x)
    print(y.shape)
