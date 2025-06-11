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