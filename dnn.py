# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

np.random.seed(0)


def gen_samples(df):
    X = df.iloc[:, :-1].as_matrix()
    Y = pd.get_dummies(df.iloc[:, -1]).as_matrix()
    return X, Y


def cal_accuracy(Y_pred, Y_true):
    correct = 0
    for y_pred, y in zip(Y_pred, Y_true):
        if y[y_pred] == 1:
            correct += 1
    return float(correct) / float(len(Y_pred))


class DNNClassifier(object):
    """
    基于多个隐藏层的BP神经网络分类器，隐层激活函数使用tanh，
    输出层使用softmax进行分类。参数更新使用批量梯度下降。

    Parameters
    ----------
    laysers_dim : list
        网络各层的维度[input, hidden,..., hidden, output]
    epochs : int
        模型训练迭代次数
    learning_rate: float
        学习率
    """
    def __init__(self, layers_dim, epochs=10000, learning_rate=0.001):
        self.laysers_dim = layers_dim
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.b = []
        self.W = []

        # 随机初始化参数
        for i in range(len(layers_dim)-1):
            self.W.append(np.random.randn(layers_dim[i], layers_dim[i+1]) / np.sqrt(layers_dim[i]))
            self.b.append(np.random.randn(layers_dim[i+1]).reshape(1, layers_dim[i+1]))

    def calculate_softmax(self, X):
        exp_scores = np.exp(X)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def calculate_loss(self, X, Y):
        input = X
        for i in range(len(self.W) - 1):
            net = np.dot(input, self.W[i]) + self.b[i]
            input = np.tanh(net)

        net = np.dot(input, self.W[-1]) + self.b[-1]
        probs = self.calculate_softmax(net)
        correct_probs = map(lambda x: np.dot(x[0], x[1].transpose()), zip(probs, Y))
        corect_logprobs = -np.log(correct_probs)
        data_loss = np.sum(corect_logprobs)

        return 1./input.shape[0] * data_loss

    def predict(self, X):
        input = X
        for i in range(len(self.W) - 1):
            net = np.dot(input, self.W[i]) + self.b[i]
            input = np.tanh(net)

        net = np.dot(input, self.W[-1]) + self.b[-1]
        probs = self.calculate_softmax(net)
        return np.argmax(probs, axis=1)

    def fit(self, X, Y, print_loss=True):
        for epoch in range(self.epochs):
            # 前向运算
            input = X
            forward = [input]
            for i in range(len(self.W) - 1):
                net = np.dot(input, self.W[i]) + self.b[i]
                input = np.tanh(net)
                forward.append(input)

            net = np.dot(input, self.W[-1]) + self.b[-1]
            probs = self.calculate_softmax(net)

            # 反向传播
            # 输出层参数更新
            dnet = probs - Y
            db = np.dot(np.ones((1, dnet.shape[0]), dtype=np.float64), dnet)
            dW = np.dot(np.transpose(forward[-1]), dnet)
            dtanh = np.dot(dnet, np.transpose(self.W[-1]))
            self.b[-1] += -self.learning_rate * db
            self.W[-1] += -self.learning_rate * dW

            # 隐层和输入层参数更新
            for i in range(len(forward) - 2, 0, -1):
                dnet = (1.0 - np.square(forward[i])) * dtanh
                db = np.dot(np.ones((1, dnet.shape[0]), dtype=np.float64), dnet)
                dW = np.dot(np.transpose(forward[i - 1]), dnet)
                dtanh = np.dot(dnet, np.transpose(self.W[i - 1]))

                # 梯度下降
                self.b[i-1] += -self.learning_rate * db
                self.W[i-1] += -self.learning_rate * dW

            # 打印loss
            if print_loss and epoch % 1000 == 0:
                print("Loss after iteration %i: %f" %(epoch, self.calculate_loss(X, Y)))


if __name__ == '__main__':
    # 读取Iris Dataset
    df = pd.read_table('bezdekIris.data.txt', sep=',', header=None)

    # 生成训练集和测试集
    df = df.sample(frac=1)
    split_pos = int(df.shape[0] * 0.8)
    df_train = df.iloc[:split_pos, :]
    df_test = df.iloc[split_pos:, :]
    X_train, Y_train = gen_samples(df_train)
    X_test, Y_test = gen_samples(df_test)

    # 构建两个维度为8的隐层
    layers_dim = [4, 8, 8, 3]

    model = DNNClassifier(layers_dim, epochs=10000, learning_rate=0.001)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    print 'accuracy=%f' % cal_accuracy(Y_pred, Y_test)
