import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 定义数据集
def generate_data(num_samples=100):
    x = np.linspace(0, 1, num_samples) # 0-1 100个点
    y1 = np.sin(2 * np.pi * x) # 2*pi
    y2 = np.sin(10 * np.pi * x) # 10*pi
    return (torch.tensor(x, dtype=torch.float32).view(-1, 1), # x
            torch.tensor(y1, dtype=torch.float32).view(-1, 1), # y1
            torch.tensor(y2, dtype=torch.float32).view(-1, 1)) # y2

# 使用生成的数据
x, y1, y2 = generate_data()

# 使用 train_test_split 进行数据集划分
x_train, x_temp, y1_train, y1_temp = train_test_split(x, y1, test_size=0.4, random_state=42)
x_val, x_test, y1_val, y1_test = train_test_split(x_temp, y1_temp, test_size=0.5, random_state=42)

_, _, y2_train, y2_temp = train_test_split(x, y2, test_size=0.4, random_state=42)
_, _, y2_val, y2_test = train_test_split(x_temp, y2_temp, test_size=0.5, random_state=42)

# 定义神经网络模型
class FlexibleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fn):
        super(FlexibleNN, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            if activation_fn == 'ReLU':
                layers.append(nn.ReLU())
            elif activation_fn == 'Tanh':
                layers.append(nn.Tanh())
            elif activation_fn == 'Sigmoid':
                layers.append(nn.Sigmoid())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 计算相对 L2 误差
def relative_l2_error(y_true, y_pred):
    return torch.norm(y_true - y_pred) / torch.norm(y_true) 

# 训练和评估模型
def train_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test, hidden_sizes, activation_fn, learning_rate, num_epochs):
    model = FlexibleNN(input_size=1, hidden_sizes=hidden_sizes, output_size=1, activation_fn=activation_fn)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # 添加 L2 正则化

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs): # 训练模型
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step() # 更新参数

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val) # 计算验证集损失
            val_losses.append(val_loss.item())

        if (epoch + 1) % 500 == 0: # 每500步输出训练和验证集的损失
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    model.eval()# 评估模型
    with torch.no_grad():
        y_pred = model(x_test)
        error = relative_l2_error(y_test, y_pred)
        print(f'Hidden Sizes: {hidden_sizes}, Activation: {activation_fn}, Learning Rate: {learning_rate}, Relative L2 Error: {error.item()}')

    return train_losses, val_losses, y_pred, error.item()

# 绘制损失曲线
def plot_losses(train_losses, val_losses, title):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.show()

# 绘制真实值与预测值对比图
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.numpy(), label='True Values')
    plt.plot(y_pred.numpy(), label='Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.title(title)
    plt.show()

# 选择一种参数配置
hidden_sizes = [64, 64]  # 隐藏层大小
activation_fn = 'ReLU'   # 激活函数
learning_rate = 0.001    # 学习率

# 对 y1 = sin(2πx) 进行训练和评估
train_losses, val_losses, y_pred1, error1 = train_and_evaluate(x_train, y1_train, x_val, y1_val, x_test, y1_test, hidden_sizes, activation_fn, learning_rate, num_epochs=5000)
print(f'Configuration: Hidden Sizes: {hidden_sizes}, Activation: {activation_fn}, Learning Rate: {learning_rate}, Relative L2 Error (y1): {error1}')
plot_losses(train_losses, val_losses, f'Loss Curves (y1) - Hidden Sizes: {hidden_sizes}, Activation: {activation_fn}, LR: {learning_rate}')
plot_predictions(y1_test, y_pred1, f'True vs Predicted (y1) - Hidden Sizes: {hidden_sizes}, Activation: {activation_fn}, LR: {learning_rate}')

# 对 y2 = sin(10πx) 进行训练和评估
train_losses, val_losses, y_pred2, error2 = train_and_evaluate(x_train, y2_train, x_val, y2_val, x_test, y2_test, hidden_sizes, activation_fn, learning_rate, num_epochs=5000)
print(f'Configuration: Hidden Sizes: {hidden_sizes}, Activation: {activation_fn}, Learning Rate: {learning_rate}, Relative L2 Error (y2): {error2}')
plot_losses(train_losses, val_losses, f'Loss Curves (y2) - Hidden Sizes: {hidden_sizes}, Activation: {activation_fn}, LR: {learning_rate}')
plot_predictions(y2_test, y_pred2, f'True vs Predicted (y2) - Hidden Sizes: {hidden_sizes}, Activation: {activation_fn}, LR: {learning_rate}')
