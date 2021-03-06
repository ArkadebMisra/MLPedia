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
            i = np.random.randint(N)
            Xt =  X[:, i:i+1]
            yt =  y[:, i:i+1]
            th_old = self.th.copy()
            th0_old = self.th0.copy()
            self.th = th_old - lrate*self.d_ridge_d_th(th_old, th0_old, Xt, yt, lam)
            self.th0 = th0_old - lrate*self.d_ridge_d_th0(th_old, th0_old, Xt, yt)
        #return th, th0



    def run_regression(self, X, y, iters=10000, lrate=0.005, lam=0):
        degree = self.degree
        X = self.x_transform(X, degree)
        return self.lin_reg(X, y, iters, lrate, lam)

    def predict(self, X):
        degree = self.degree
        X = self.x_transform(X, degree)
        return self.weighted_sum(self.th, self.th0, X)


    def r2_score(self, X, y):
        guess = self.predict(X)
        return 1 - (np.sum((guess-y)**2)/
                    np.sum((y-np.mean(y))**2))


