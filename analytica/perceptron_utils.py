import numpy as np


class Perceptron:
    def __init__(self):
        self.th = None
        self.th0 = None

    def positive(self, x, th, th0):
        '''
        x is dimension d by 1
        th is dimension d by 1
        th0 is dimension 1 by 1
        return 1 by 1 matrix of +1, 0, -1
        '''
        return np.sign(np.dot(np.transpose(th), x) + th0)

    def score(self, data, labels, th, th0):
        '''
        data is dimension d by n
        labels is dimension 1 by n
        ths is dimension d by 1
        th0s is dimension 1 by 1
        return 1 by 1 matrix of integer indicating number of data points correct for
        each separator.
        '''
        return np.sum(self.positive(data, th, th0) == labels)
    

    def averaged_perceptron(self, data, labels, T):
        (d, n) = data.shape

        theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
        theta_sum = theta.copy() 
        theta_0_sum = theta_0.copy()
        for t in range(T):
            for i in range(n):
                x = data[:,i:i+1]
                y = labels[:,i:i+1]
                if y * self.positive(x, theta, theta_0) <= 0.0:
                    theta = theta + y * x
                    theta_0 = theta_0 + y
                theta_sum = theta_sum + theta
                theta_0_sum = theta_0_sum + theta_0
        theta_avg = theta_sum / (T*n)
        theta_0_avg = theta_0_sum / (T*n)
        return theta_avg, theta_0_avg

    def run_perceptron(self, data, labels, T=10000):
        self.th, self.th0 = self.averaged_perceptron(data, labels, T=T)

    def predict_label(self, x):
        label =  self.positive(x, self.th, self.th0)
        return np.where(label>=0,1,-1)

    def get_accuracy(self, X, y):
        return self.score(X, y, self.th, self.th0)/X.shape[1]





