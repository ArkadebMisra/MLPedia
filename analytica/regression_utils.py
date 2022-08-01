import numpy as np

class RegressionModel:

    #d is feature dimention/weight dimention
    def __init__(self, d,degree=1):
        self.degree = degree
        self.D = d*degree
        self.th = np.random.normal(0, 1.0 * self.D ** (-.5), [self.D, 1])
        self.th0 = np.zeros((1, 1))

    

    # In all the following definitions:
    # x is d by n : input data
    # y is 1 by n : output regression values
    # th is d by 1 : weights
    # th0 is 1 by 1 or scalar
    def weighted_sum(self, th, th0, X):
        """ Returns the predicted y: 1*n"""
        # print(th.shape)
        # print(X.shape)
        # print(th0.shape)
        y = np.dot(th.T, X) + th0
        return y

    def square_loss(self, th, th0, X, y):
        '''Returns the squared loss between y_pred and y'''
        return (self.weighted_sum(th, th0, X) -y )**2

    def mean_square_loss(self, th, th0, X, y):
        '''Return the mean squared loss between y_pred and y'''
        return np.mean(self.square_loss(th, th0, X, y), axis = 1, keepdims = True)

    #####################gradients###################
    def ridge_obj(self, th, th0, X, y, lam):
        return self.mean_square_loss(th, th0, X, y) + lam*(np.linalg.norm(th)**2)

    def d_ridge_d_th(self, th, th0, X, y, lam):
        return 2*np.mean((self.weighted_sum(th, th0, X) -y )*X, axis=1, keepdims=True) + 2*lam*th

    def d_ridge_d_th0(self, th, th0, X, y):
        return 2*np.mean((self.weighted_sum(th, th0, X) -y ), axis=1, keepdims=True)


    def x_transform(self, X, degree):
        
        # X --> Input.
        # degrees --> A list, We add X^(value) feature to the input
        #             where value is one of the values in the list.
        
        # making a copy of X.
        t = X.copy()
        
        # Appending columns of higher degrees to X.
        for i in range(2, degree+1):
            X = np.append(X, t**i, axis=0)
                
        return X
    

    def lin_reg(self, X, y, iters=10000, lrate=0.005, lam=.1):
        D, N = self.D, X.shape[1]
        # th = np.random.normal(0, 1.0 * D ** (-.5), [D, 1])
        # th0 = np.zeros((1, 1))
        for it in range(iters):
            # print(it)
            i = np.random.randint(N)
            Xt =  X[:, i:i+1]
            yt =  y[:, i:i+1]
            th_old = self.th.copy()
            th0_old = self.th0.copy()
            self.th = th_old - lrate*self.d_ridge_d_th(th_old, th0_old, Xt, yt, lam)
            self.th0 = th0_old - lrate*self.d_ridge_d_th0(th_old, th0_old, Xt, yt)
            #trying to fix overflow(bad fix)
            # self.th = np.around(self.th, decimals=2)
            # self.th0 = np.around(self.th0, decimals=2)
            # print(self.th, self.th0)
        #return th, th0

    #closed form formula for linear regression
    #gradient descent producing overflow error
    def closed_form_lin_reg(self, X, Y, iters=10000, lrate=0.005, lam=.1):
        """
        Computes the closed form solution of linear regression with L2 regularization

        Args:
            X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
            Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
                data point
            lambda_factor - the regularization constant (scalar)
        Returns:
            theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
            represents the y-axis intercept of the model and therefore X[0] = 1
        """
        d, n = X.shape
        X = np.vstack((X, np.ones(X.shape[1]))).T
        Y=Y.T
        I = np.identity(X.shape[1])
        theta = np.linalg.inv(X.T @ X + lam * I) @ X.T @ Y
        # print(theta.shape)
        # print(theta)
        self.th = theta[:d, :]
        self.th0 = theta[d:, :]
        # print(self.th.shape)
        # print(self.th0.shape)

    def run_regression(self, X, y, iters=10000, lrate=0.005, lam=.1):
        degree = self.degree
        X = self.x_transform(X, degree)
        # gradient decent lin reg not working
        # return self.lin_reg(X, y, iters, lrate, lam)
        
        # trying closed form
        return self.closed_form_lin_reg(X, y, iters, lrate, lam)

    def predict(self, X):
        degree = self.degree
        X = self.x_transform(X, degree)
        return self.weighted_sum(self.th, self.th0, X)


    def r2_score(self, X, y):
        guess = self.predict(X)
        return 1 - (np.sum((guess-y)**2)/
                    np.sum((y-np.mean(y))**2))


