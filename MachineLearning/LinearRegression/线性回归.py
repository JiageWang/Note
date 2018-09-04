import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class LinearRegression(object):
    def __init__(self):
        self.theta = None
        self.loss_list = []

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
            y_pred = X @ theta
            error = y_pred - y
            loss = np.sum(error**2) / (2 * self.sample_num)
            print("At iter {0}, loss = {1}".format(i + 1, loss))
            self.loss_list.append(loss)

            # upgrade theta through gradient descent
            grad = X.T @ error / self.sample_num
            theta = theta - lr * grad

        # record the final theta
        self.theta = theta
        print("Final theta: {0}".format(theta))

    def predict(self, X):
        return X @ self.theta

    def plot_data(self):
        """plot the data distribute and the decision plane"""
        fig = plt.figure()
        if self.feature_num == 1:
            ax = fig.add_subplot(111)
            ax.scatter(self.X[:, 0], self.y[:, 0])
            x_ = np.array([self.X.min(), self.X.max()])
            y_ = self.theta[0] * x_ + self.theta[1]
            ax.plot(x_, y_, label="decision plane", c='r')
            plt.title("Data distribution")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.show()
        elif self.feature_num == 2:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.X[:, 0], self.X[:, 1], self.y[:, 0])
            x_ = np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), 100)
            y_ = np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), 100)
            x_, y_ = np.meshgrid(x_, y_)
            z_ = self.theta[0, 0] * x_ + self.theta[1, 0] * y_
            ax.plot_surface(x_, y_, z_)
            plt.show()

        else:
            print("unable to show data in high dimentional space")

    def plot_loss(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_ = range(len(self.loss_list))
        y_ = self.loss_list
        ax.plot(x_, y_)
        plt.xlabel("iters")
        plt.ylabel("loss")
        plt.show()

    # def plot_theta(self):
    #     if self.feature_num == 1:
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
    #         w = self.theta[0, 0]
    #         b = self.theta[1, 0]
    #         range_w = np.linspace(w - 5, w + 5, 100)
    #         range_b = np.linspace(b - 5, b + 5, 100)
    #         w_, b_ = np.meshgrid(range_w, range_b)
    #         ax.plot_surface(w_, b_, loss)
    #         plt.show()


if __name__ == "__main__":
    # read data from txt
    data = pd.read_csv('./ex1data1.txt', header=None)
    #data = pd.read_csv('./ex1data2.txt', header=None)
    data = (data - data.mean()) / data.std()
    data = data.values
    X = data[:, :-1]
    y = data[:, -1:]
    print(X.shape)
    print(y.shape)

    # train model
    lg = LinearRegression()
    lg.fit(X, y, lr=0.01, iters=1000)
    lg.plot_data()
    lg.plot_loss()
    # lg.plot_theta()
