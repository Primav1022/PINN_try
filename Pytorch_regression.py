# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:40:07 2025

@author: WeiZh
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 生成数据
x = torch.linspace(0, 1, 100).reshape(-1, 1) # 0-1 100个点
y = torch.sin(2 * np.pi * x) # 2*pi

# 拆分为训练集、验证集和测试集
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(1, 64)  # 隐藏层
        self.output = nn.Linear(64, 1)  # 输出层

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # ReLU激活函数
        return self.output(x)

# 初始化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 训练模型
num_epochs = 5000
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # 清除梯度
    output_train = model(x_train)  # 前向传播
    loss_train = criterion(output_train, y_train)  # 计算训练集损失
    loss_train.backward()  # 反向传播
    optimizer.step()  # 更新参数
    
    # 计算验证集损失
    model.eval()
    with torch.no_grad():
        output_val = model(x_val)
        loss_val = criterion(output_val, y_val)
    
    # 保存训练集和验证集的损失
    train_losses.append(loss_train.item())
    val_losses.append(loss_val.item())

    # 每500步输出训练和验证集的损失
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}')

# 绘制训练集和验证集的损失曲线
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.show()

# 测试模型
model.eval()
with torch.no_grad():
    predicted_test = model(x_test).detach()


# 绘制结果
plt.plot(x.numpy(), y.numpy(), label='True value')
plt.scatter(x_test.numpy(), y_test.numpy(), label='Test value', color='blue', s=20)
plt.scatter(x_test.numpy(), predicted_test.numpy(), label='Predicted value', color='red', s=20)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Test Set Prediction')
plt.show()
