import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def painting(training_inputs, labels, test_inputs, perceptron):
    # 绘制训练数据点
    plt.figure(figsize=(8, 6))
    plt.scatter(training_inputs[labels == 1][:, 0], training_inputs[labels == 1][:, 1], color='blue', label='Apf')
    plt.scatter(training_inputs[labels == 0][:, 0], training_inputs[labels == 0][:, 1], color='red', label='Af')

    # 绘制测试数据点
    plt.scatter(test_inputs[:, 0], test_inputs[:, 1], color='green', marker='s', label='测试数据')

    # 绘制决策边界
    slope = -perceptron.weights[0] / perceptron.weights[1]
    intercept = -perceptron.bias / perceptron.weights[1]
    x_vals = np.linspace(1.1, 1.6, 100)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='black', linestyle='dashed', label='决策边界（分类面）')

    plt.xlabel('触角长度')
    plt.ylabel('翅膀长度')
    plt.legend()
    plt.title('感知器分类器')
    plt.grid(True)
    plt.show()