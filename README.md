# Python基础精要
# 变量与作用域
核心类型：int/float/str/bool/list/tuple/dict/set

# 作用域控制：
global_var = 10
def func():
    global global_var  # 声明全局变量
    nonlocal outer_var  # 声明外层非全局变量（嵌套函数）
类型转换：int("1") → 1, str(3.14) → "3.14", list(range(3)) → [0,1,2]

# 运算符全解
类别	运算符
算术	+ - * / //(整除) %(取模) **(幂)
比较	== != > < >= <= is(对象ID比较)
逻辑	and or not
位运算	&(与) `	(或) ^(异或) ~(取反) <<(左移) >>`(右移)
# 控制流语句
* 条件分支：
if x > 0: 
    print("正数")
elif x == 0:
    print("零")
else:
    print("负数")
* 循环控制：
for i in range(5): 
    if i == 3: continue  # 跳过当前迭代
    print(i)

while True:
    if condition: break  # 终止循环
# 异常处理：
try:
    risky_operation()
except ValueError as e:
    print(f"值错误: {e}")
finally:
    cleanup_resources()  # 始终执行
函数进阶
参数类型：

def func(a, b=2, *args, **kwargs):
    # a:必选参数, b:默认参数, args:元组, kwargs:字典
    print(a, b, args, kwargs)
Lambda函数：add = lambda x,y: x+y → add(3,5) 返回 8

高阶函数：

python
def apply(func, data):
    return [func(x) for x in data]

apply(lambda x: x**2, [1,2,3])  # 返回 [1,4,9]
# 模块化管理
导入方式：


import os  # 全模块导入
from math import sqrt  # 精确导入
import pandas as pd  # 别名导入
包结构：

text
my_package/
    __init__.py  # 包标识文件
    module1.py
    subpackage/
常用三方库：requests(HTTP)、numpy(数值计算)、pandas(数据分析)

Git操作速查表
bash
# 初始化与基础操作
git init
git add .  # 添加所有修改到暂存区
git commit -m "提交说明"

# 远程仓库管理
git remote add origin https://github.com/user/repo.git
git pull --rebase origin main  # 变基式拉取（避免合并提交）
git push -u origin main  # -u设置默认上游分支

# 全局配置
git config --global user.name "用户名"
git config --global user.email "邮箱"
PyTorch环境配置指南
通过Anaconda安装
bash
# 创建虚拟环境
conda create -n pytorch python=3.9 -y

# 激活环境
conda activate pytorch

# 配置清华镜像源（加速下载）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes

# 安装PyTorch（GPU版本）
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
验证安装

import torch
print(torch.__version__)            # 查看版本（应≥1.12）
print(torch.cuda.is_available())    # 检查GPU是否可用
注意事项：
确保NVIDIA驱动支持CUDA 11.7（通过nvidia-smi查看）
安装前关闭VPN避免网络干扰

## 深度学习基础
训练数据集一定是两次循环
欠拟合：训练训练数据集表现不好，验证表现不好
过拟合：训练数据训练过程表现得很好，在我得验证过程表现不好

# 卷积神经网络
import torch
import torch.nn.functional as F
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

# 不满足conv2d的尺寸要求
print(input.shape)
print(kernel.shape)

# 尺寸变换
input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
print(input.shape)
print(kernel.shape)

output = F.conv2d(input=input,weight=kernel,stride=1)
print(output)

output2 = F.conv2d(input=input,weight=kernel,stride=2)
print(output2)

# padding 在周围扩展一个像素，默认为0；
output3 = F.conv2d(input=input,weight=kernel,stride=1,padding=1)
print(output3)

## 卷积神经网络

这段代码的作用只是为了拿到我的conv_logs里面的文件

使用tensorboard命令打开
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)


class CHEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=3,
                               stride=1,
                               padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


chen = CHEN()
print(chen)

writer = SummaryWriter("conv_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = chen(imgs)

    # print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    # print(output.shape)  # torch.Size([64, 6, 30, 30])
    writer.add_images("input", imgs, step)

    # torch.Size([64, 6, 30, 30]) ->([**, 3, 30, 30])
    output = torch.reshape(output, (-1, 3, 30, 30))  # -1:会根据后面的值进行调整
    writer.add_images("output", output, step)
    step += 1

定义我们的网络模型


# tensorboard --logdir=conv_logs 要使用绝对路径
我的路径即为：tensorboard --logdir=G:\py\pythonProjectdemo\day2\conv_logs

# 池化层
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#
dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)

/*最大池化没法对long整形进行池化
input = torch.tensor([[1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                      [2,1,0,1,1]], dtype = torch.float)
input =torch.reshape(input,(-1,1,5,5)) 
print(input.shape)
*/

class Chen(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_1 = MaxPool2d(kernel_size=3,
                                   ceil_mode=False)
    def forward(self,input):
        output = self.maxpool_1(input)
        return output

chen = Chen()

writer = SummaryWriter("maxpool_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs,step)
    output = chen(imgs)
    writer.add_images("ouput",output,step)
    step += 1
writer.close()


//output = chen(input)
//print(output)