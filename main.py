import numpy as np

from draw import painting


# 定义感知器模型
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels, epochs):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights += (label - prediction) * inputs
                self.bias += (label - prediction)


# 训练数据集（触角长度、翅膀长度）
training_inputs = np.array(
    [[1.24, 1.27], [1.36, 1.74], [1.38, 1.82], [1.38, 1.64], [1.38, 1.90], [1.40, 1.70], [1.48, 1.82], [1.54, 1.82],
     [1.56, 2.08], [1.14, 1.82], [1.18, 1.96], [1.20, 1.86], [1.26, 2.00], [1.28, 2.00], [1.30, 1.96]])
labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # 类别标签（1为Apf,0为Af）

# 初始化感知器
perceptron = Perceptron(input_size=2)

# 训练感知器模型
perceptron.train(training_inputs, labels, epochs=1000)

# 计算每个训练样本的预测类别
predicted_labels = np.array([perceptron.predict(inputs) for inputs in training_inputs])

# 输出解向量
print("解向量（权重向量）:", perceptron.weights)
# 检查是否所有样本都被正确分类
if np.array_equal(predicted_labels, labels):
    print("解向量可以将原始样本进行类别区分。")
else:
    print("解向量不能完全将原始样本进行类别区分。")

# 测试数据集
test_inputs = np.array([[1.24, 1.80],
                        [1.28, 1.84],
                        [1.40, 2.04]])
# 预测测试数据集的类别
real_result = [1, 1, 0]
right = 0
for i in range(len(test_inputs)):
    prediction = perceptron.predict(test_inputs[i])
    if prediction == 1:
        print(f'预测{test_inputs[i]} 属于 Apf')
    else:
        print(f'预测{test_inputs[i]} 属于 Af')
    if real_result[i] == 1:
        print('实际属于 Apf')
    else:
        print('实际属于 Af')
    if prediction == real_result[i]:
        right += 1

# 计算准确率
accuracy = (right / len(test_inputs)) * 100
print(f'准确率: {accuracy:.2f}%')

# 绘制分类结果图
painting(training_inputs, labels, test_inputs, perceptron)
