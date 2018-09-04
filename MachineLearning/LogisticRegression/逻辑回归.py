import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class LogisticRegression(object):
    def __init__(self):
        self.theta = None
        self.loss_list = []

    def sigmoid(self, z):
        return 1.0/(1 + np.exp(-z))

    def loss(self, z, y):
        return ((-y.T @ np.log(self.sigmoid(z)) - (1 - y).T @ np.log(1 - self.sigmoid(z)))/(self.sample_num *2))/self.sample_num

    def fit(self, X, y, lr=0.001, iters=1000):
        """train the model with input X and y"""
        # add bias if X without bias
        if np.sum(X[:, -1] - np.ones(X.shape[0])) != 0:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        self.X = X
        self.y = y
        self.sample_num = self.X.shape[0]
        self.feature_num = self.X.shape[1] - 1
        theta = np.random.randn(self.X.shape[1], 1)

        for i in range(iters):
            # compute and record the loss
            z = X @ theta
            error = self.sigmoid(z) - self.y
            loss = self.loss(z, self.y)[0,0]
            print("At iter {0}, loss = {1}".format(i + 1, loss))
            self.loss_list.append(loss)

            # upgrade theta through gradient descent
            grad = X.T @ error
            theta = theta - lr * grad /self.sample_num

        # record the final theta
        self.theta = theta
        print("Final theta: {0}".format(theta))

    def predict(self, X):
        return X @ self.theta

    def plot_data(self):
        """plot the data distribute and the decision plane"""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        positive_index = np.where(self.y==1)[0]
        negative_index = np.where(self.y==0)[0]
        x_positive = self.X[positive_index, 0]
        y_positive = self.X[positive_index, 1]
        x_negative = self.X[negative_index, 0]
        y_negative = self.X[negative_index, 1]
        ax.scatter(x_positive,y_positive,c='r')
        ax.scatter(x_negative,y_negative,c='b')
        x_ = np.linspace(self.X[:,0].min(),self.X[:, 1].max(),100)
        theta =self.theta
        y_ = theta[2,0]/theta[1,0]- x_*theta[0,0]/theta[1,0]
        ax.plot(x_, y_)

        plt.show()



    def plot_loss(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_ = range(len(self.loss_list))
        y_ = self.loss_list
        ax.plot(x_, y_)
        plt.xlabel("iters")
        plt.ylabel("loss")
        plt.show()



if __name__ == "__main__":
    # read data from txt
    data = pd.read_csv('./ex2data1.txt', header=None)
    #data = pd.read_csv('./ex1data2.txt', header=None)
    data = data.values
    X = data[:, :-1]
    y = data[:, -1:]
    print(X.shape)
    print(y.shape)

    # train model
    lg = LogisticRegression()
    lg.fit(X, y)

    lg.plot_data()
    lg.plot_loss()
    # lg.plot_theta()
