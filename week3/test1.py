"""
浅层神经网络模型实现：只有一个隐藏层
实现思路：
    1.定义网络的各层维度
    2.根据各层维度，初始化参数
    3.循环实现前向反馈神经网络
    4.根据梯度下降法实现反向传播神经网络，不断更新参数,包括：
        计算代价函数
        利用反向传播，计算参数的梯度
        利用梯度下降，更新参数
    5.训练模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 创建数据集，生成花朵形状，两个类别
def load_data_sets():
    np.random.seed(1)  # np.random.seed(n)函数用于生成指定随机数
    num = 400  # 总样本数
    class_simple = int(num / 2)  # 每个类别的样本数
    dimen = 2  # 样本维度为2
    x = np.zeros((num, dimen))  # 样本特征，初始化全0.(400, 2)
    y = np.zeros((num, 1), dtype='uint8')  # 标签(400, 1)
    radius = 4  # 花朵半径
    # 生成数据点
    for j in range(2):  # 生成两类数据:0/1
        ix = range(class_simple * j, class_simple * (j + 1))  # 当前类别的索引范围
        t = np.linspace(j * 3.12, (j + 1) * 3.12, class_simple) + np.random.randn(
            class_simple) * 0.2  # 生成角度 t，并添加一些随机噪声
        r = radius * np.sin(4 * t) + np.random.randn(class_simple) * 0.2  # 计算半径 r，使用正弦波形生成花瓣效果，并添加噪声
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]  # 将极坐标转为笛卡尔坐标（x1, x2）
        y[ix] = j
    return x, y


x, y = load_data_sets()
# 可视化数据
plt.figure(figsize=(8, 6))
plt.scatter(x[y.flatten() == 0, 0], x[y.flatten() == 0, 1], color='r', label='Class 0')  # Class 0 (红色)
plt.scatter(x[y.flatten() == 1, 0], x[y.flatten() == 1, 1], color='b', label='Class 1')  # Class 1 (蓝色)

plt.title('Generated Dataset with Two Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


def sigmoid(x):
    x = x.astype(np.float64)  # 强制转换类型
    return 1 / (1 + np.exp(-x))


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


# 计算各层维度
def get_network_dimension(x, y):
    input_dimension = x.shape[0]  # 输入层的神经元数
    hidden_dimension = 4  # 隐藏层的神经元数
    output_dimension = y.shape[0]  # 输出层的神经元数
    return input_dimension, hidden_dimension, output_dimension


# 初始化参数
def init_params(input_dim, hidden_dim, output_dim):
    np.random.seed(2)
    # 使用正态分布生成小的随机值作为初始参数
    # randn(m,n)返回一个m*n的随机项矩阵，具有标准正态分布。
    w1 = np.random.randn(hidden_dim, input_dim) * 0.01
    b1 = np.zeros((hidden_dim, 1))
    w2 = np.random.randn(output_dim, hidden_dim) * 0.01
    b2 = np.zeros((output_dim, 1))
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return params


# 前向反馈神经网络
def forward_propagation(x, params):
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']
    z1 = np.dot(w1, x) + b1  # 隐藏层的加权输入
    A1 = tanh(z1)  # 隐藏层的激活输出
    z2 = np.dot(w2, A1) + b2  # 输出层的加权输入
    A2 = tanh(z2)  # 输出层的激活输出
    param = {'z1': z1, 'A1': A1, 'z2': z2, 'A2': A2}
    return A2, param


# 计算代价函数
def compute_cost(predict, label):
    m = label.shape[1]  # 样本数量
    predict = np.clip(predict, 1e-8, 1 - 1e-8)  # 防止log(0)产生无效值
    # 计算交叉熵损失
    log_loss = np.multiply(label, np.log(predict)) + np.multiply((1 - label), np.log(1 - predict))
    # 代价函数
    loss = -1 / m * np.sum(log_loss)
    loss = np.squeeze(loss)
    return loss


# 反向传播
def backward(params, cache, x, y):
    m = x.shape[1]
    W1 = params["w1"]
    W2 = params["w2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2 - y  # 输出层的梯度
    dW2 = (1 / m) * np.dot(dZ2, A1.T)  # 计算 W2 的梯度
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # 计算 b2 的梯度
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))  # 隐藏层的梯度
    dW1 = (1 / m) * np.dot(dZ1, x.T)  # 计算 W1 的梯度
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  # 计算 b1 的梯度
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads


# 更新参数
def update(params, grads, learning_rate=0.1):
    # 从params字典中获取当前的权重和偏置
    W1 = params["w1"]
    b1 = params["b1"]
    W2 = params["w2"]
    b2 = params["b2"]
    # 从grads字典中获取当前的梯度
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    # 更新参数（权重和偏置）
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    # 返回更新后的参数
    updated_params = {
        "w1": W1,
        "b1": b1,
        "w2": W2,
        "b2": b2
    }
    return updated_params


# 训练神经网络
def train_nn(x, y, hidden_units, iters=10000, print_cost=False):
    costs = []
    np.random.seed(3)
    input_units, output_units = x.shape[0], y.shape[0]
    params = init_params(input_units, hidden_units, output_units)
    for i in range(iters):
        A2, param = forward_propagation(x, params)
        cost = compute_cost(A2, y)
        costs.append(cost)
        grads = backward(params, param, x, y)
        params = update(params, grads)
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost:.6f}")

    # 可视化代价函数
    plt.plot(np.arange(iters), costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function vs. Iterations')
    plt.show()

    return params


# 加载数据集
x, y = load_data_sets()

# 训练神经网络并可视化代价函数
params = train_nn(x.T, y.T, hidden_units=4, iters=10000, print_cost=True)


def plot_decision_boundary(X, Y, model_params, plot_title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 使用训练好的模型对网格数据进行预测
    A2, _ = forward_propagation(grid.T, model_params)
    predictions = (A2 > 0.5).astype(int).reshape(xx.shape)

    # 绘制背景颜色和数据点
    plt.contourf(xx, yy, predictions, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), edgecolors='k', marker='o', cmap=plt.cm.RdBu)
    plt.title(plot_title)
    plt.show()


# 可视化决策边界
plot_decision_boundary(x, y, params)


def predict(params, x):
    # 前向传播，计算预测结果 A2
    A2, cache = forward_propagation(x, params)
    # 预测值四舍五入，大于 0.5 预测为 1，否则为 0
    predictions = np.round(A2)
    return predictions


def accuracy(true_labels, predicted_labels):
    # 计算预测正确的样本数（1 正确预测的个数 + 0 正确预测的个数）
    correct_predictions = np.dot(true_labels, predicted_labels.T) + np.dot(1 - true_labels, (1 - predicted_labels).T)
    # 使用 .item() 提取标量值，避免警告
    accuracy = correct_predictions.item() / true_labels.shape[1]
    return accuracy


# 对训练集进行预测
y_pred = predict(params, x.T)

# 计算模型的准确率
accuracy_value = accuracy(y.T, y_pred)
print(f"模型的准确率为: {accuracy_value * 100:.2f}%")
