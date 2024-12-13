import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

train_data = h5py.File(
    'E:/BaiduNetdiskDownload/Improving Deep Neural Networks Hyperparameter tuning, Regularization and Optimization/week5/Regularization/datasets/train_catvnoncat.h5')
test_data = h5py.File(
    'E:/BaiduNetdiskDownload/Improving Deep Neural Networks Hyperparameter tuning, Regularization and Optimization/week5/Regularization/datasets/test_catvnoncat.h5')
print(train_data.keys())  # ['list_classes', 'train_set_x', 'train_set_y']
print(test_data.keys())  # ['list_classes', 'test_set_x', 'test_set_y']
print(train_data['list_classes'][:])  # [b'non-cat' b'cat']
print(train_data['train_set_x'].shape)  # (209, 64, 64, 3) 图像209张，大小64*64*3
print(train_data['train_set_y'].shape)  # (209,)

print(test_data['test_set_x'].shape)  # (50, 64, 64, 3)
print(test_data['test_set_y'].shape)  # (50,)

# 取出训练集
train_data_orgin = train_data['train_set_x'][:]
train_label_orgin = train_data['train_set_y'][:]
# 取出测试集
test_data_orgin = test_data['test_set_x'][:]
test_label_orgin = test_data['test_set_y'][:]

# 查看一张图像
index = np.random.randint(0, test_data_orgin.shape[0])
image = test_data_orgin[index]
plt.imshow(image)
plt.show()

# 数据维度处理
num_train = train_data_orgin.shape[0]
num_test = test_label_orgin.shape[0]
print(num_train)  # 209
print(num_test)  # 50

train_data_trans = train_data_orgin.reshape(num_train, -1).T
test_data_trans = test_data_orgin.reshape(num_test, -1).T
print(train_data_trans.shape)  # (12288, 209)        64*64*3=12288
print(test_data_trans.shape)  # (12288, 50)

# 标签处理
train_label_trans = train_label_orgin.reshape(train_label_orgin.shape[0], 1).T
print(train_label_trans.shape)  # (1, 209)
test_label_trans = test_label_orgin.reshape(test_label_orgin.shape[0], 1).T
print(test_label_trans.shape)  # (1, 50)

# 标准化数据(都在0-1之间)
train_data_sta = train_data_trans / 255
test_data_sta = test_data_trans / 255


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 初始化参数w和b
n_dim = test_data_sta.shape[0]
print(n_dim)  # 12288
w = np.zeros((n_dim, 1))
b = 0


# 前向传播
def propagate(w, b, x, y, lambda_reg=0.01, epsilon=1e-8):
    z = np.dot(w.T, x) + b
    A = sigmoid(z)

    m = x.shape[1]

    # 防止对数出现 log(0)，加入 epsilon 防止值为零
    A = np.clip(A, epsilon, 1 - epsilon)  # 将 A 限制在 [epsilon, 1-epsilon] 范围内

    # 计算代价函数，加入正则化项
    J = -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A)) + lambda_reg / (2 * m) * np.sum(np.square(w))

    # 计算梯度，加入正则化项
    dw = 1 / m * np.dot(x, (A - y).T) + (lambda_reg / m) * w
    db = 1 / m * np.sum(A - y)

    return dw, db, J


# 优化
def optimize(w, b, x, y, alpha, iters):
    costs = []
    for i in range(iters):
        dw, db, J = propagate(w, b, x, y)
        w = w - alpha * dw
        b = b - alpha * db
        if i % 100 == 0:
            costs.append(J)
            print("损失函数", J)

    return w, b, dw, db, costs


alpha = 0.05
iters = 1000
w, b, dw, db, costs = optimize(w, b, train_data_sta, train_label_trans, alpha, iters)
# 绘制损失函数的变化曲线
plt.plot(costs)
plt.xlabel('Iterations (per 100)')  # x轴：每100次迭代
plt.ylabel('Cost')  # y轴：损失值
plt.title('Loss Function During Optimization')  # 图表标题
plt.show()


# 预测
# 预测函数
def predict(w, b, x_test):
    z = np.dot(w.T, x_test) + b
    A = sigmoid(z)
    num = x_test.shape[1]
    y_pred = np.zeros((1, num))
    for i in range(num):
        if A[:, i] > 0.5:
            y_pred[:, i] = 1
        else:
            y_pred[:, i] = 0
    return y_pred


# 展示预测的图像及标签
def display_random_prediction(w, b, x_test, y_test):
    # 随机选择一个索引
    index = np.random.randint(0, x_test.shape[1])

    # 获取图像和对应的标签
    image = x_test[:, index].reshape(64, 64, 3)  # 恢复图像的原始形状
    true_label = y_test[0, index]

    # 预测标签
    predicted_label = predict(w, b, x_test)[:, index]

    # 显示图像
    plt.imshow(image)
    plt.axis('off')  # 不显示坐标轴
    plt.title(f"True Label: {'Cat' if true_label == 1 else 'Non-cat'}, "
              f"Predicted: {'Cat' if predicted_label == 1 else 'Non-cat'}")
    plt.show()


# 随机显示一张预测图像
display_random_prediction(w, b, test_data_sta, test_label_trans)


# 计算正确率
def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


# 训练集和测试集的预测
train_pred = predict(w, b, train_data_sta)
test_pred = predict(w, b, test_data_sta)

# 计算训练集和测试集的准确率
train_acc = accuracy(train_pred, train_label_trans)
test_acc = accuracy(test_pred, test_label_trans)
print(f"训练集准确率: {train_acc * 100:.2f}%")
print(f"测试集准确率: {test_acc * 100:.2f}%")
