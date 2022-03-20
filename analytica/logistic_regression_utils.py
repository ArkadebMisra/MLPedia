from matplotlib.pyplot import axis
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors





def rv(value_list):
    return np.array([value_list])


def cv(value_list):
    return np.transpose(rv(value_list))
    

######################################################################
# Plotting

def tidy_plot(xmin, xmax, ymin, ymax, center = False, title = None,
                 xlabel = None, ylabel = None):
    '''
    Set up axes for plotting
    xmin, xmax, ymin, ymax = (float) plot extents
    Return matplotlib axes
    '''
    #plt.ion()
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
    eps = .05
    plt.xlim(xmin-eps, xmax+eps)
    plt.ylim(ymin-eps, ymax+eps)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def plot_separator(ax, th, th_0):
    '''
    Plot separator in 2D
    ax = (matplotlib plot) plot axis
    th = (numpy array) theta
    th_0 = (float) theta_0
    '''
    xmin, xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    pts = []
    eps = 1.0e-6
    # xmin boundary crossing is when xmin th[0] + y th[1] + th_0 = 0
    # that is, y = (-th_0 - xmin th[0]) / th[1]
    if abs(th[1,0]) > eps:
        pts += [np.array([x, (-th_0 - x * th[0,0]) / th[1,0]], dtype='float64') \
                                                        for x in (xmin, xmax)]
    if abs(th[0,0]) > 1.0e-6:
        pts += [np.array([(-th_0 - y * th[1,0]) / th[0,0], y], dtype='float64') \
                                                         for y in (ymin, ymax)]
    in_pts = []
    for p in pts:
        if (xmin-eps) <= p[0] <= (xmax+eps) and \
           (ymin-eps) <= p[1] <= (ymax+eps):
            duplicate = False
            for p1 in in_pts:
                if np.max(np.abs(p - p1)) < 1.0e-6:
                    duplicate = True
            if not duplicate:
                in_pts.append(p)
    if in_pts and len(in_pts) >= 2:
        # Plot separator
        vpts = np.vstack(in_pts)
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Plot normal
        vmid = 0.5*(in_pts[0] + in_pts[1])
        scale = np.sum(th*th)**0.5
        diff = in_pts[0] - in_pts[1]
        dist = max(xmax-xmin, ymax-ymin)        
        vnrm = vmid + (dist/10)*(th.T[0]/scale)
        vpts = np.vstack([vmid, vnrm])
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Try to keep limits from moving around
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
    else:
        print('Separator not in plot range')

def plot_data(data, labels, ax = None, clear = False,
                  xmin = None, xmax = None, ymin = None, ymax = None):
    '''
    Make scatter plot of data.
    data = (numpy array)
    ax = (matplotlib plot)
    clear = (bool) clear current plot first
    xmin, xmax, ymin, ymax = (float) plot extents
    returns matplotlib plot on ax 
    '''
    if ax is None:
        if xmin == None: xmin = np.min(data[0, :]) - 0.5
        if xmax == None: xmax = np.max(data[0, :]) + 0.5
        if ymin == None: ymin = np.min(data[1, :]) - 0.5
        if ymax == None: ymax = np.max(data[1, :]) + 0.5
        ax = tidy_plot(xmin, xmax, ymin, ymax)

        x_range = xmax - xmin; y_range = ymax - ymin
        if .1 < x_range / y_range < 10:
            ax.set_aspect('equal')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    elif clear:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    colors = np.choose(labels > 0, cv(['r', 'g']))[0]
    ax.scatter(data[0,:], data[1,:], c = colors,
                    marker = 'o', s=50, edgecolors = 'none')
    # Seems to occasionally mess up the limits
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, which='both')
    #ax.axhline(y=0, color='k')
    #ax.axvline(x=0, color='k')
    return ax

def plot_nonlin_sep(predictor, ax = None, xmin = None , xmax = None,
                        ymin = None, ymax = None, res = 30):
    '''
    Must either specify limits or existing ax
    Shows matplotlib plot on ax
    '''
    if ax is None:
        ax = tidy_plot(xmin, xmax, ymin, ymax)
    else:
        if xmin == None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

    cmap = colors.ListedColormap(['black', 'white'])
    bounds=[-2,0,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)            
            
    ima = np.array([[predictor(x1i, x2i) \
                         for x1i in np.linspace(xmin, xmax, res)] \
                         for x2i in np.linspace(ymin, ymax, res)])
    im = ax.imshow(np.flipud(ima), interpolation = 'none',
                       extent = [xmin, xmax, ymin, ymax],
                       cmap = cmap, norm = norm)

######################################################################



class LogisticRegression:

    def __init__(self):
        self.th = None
        self.th0 = None

    #z is (1 x b) numpy array
    #returns a (1 x b) numpy array, sigmoid function applied to each element
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))



    #X is the batch of input. (m X b) numpy array
    #Th is the weight of LR model. (m X 1) numpy array
    #th0 is the bias of the LR model (1 X b ) numpy array
    #computes z = th^x + th0
    #returns z (1xb) numpy array
    def weighted_sum(self, th, th0, X):
        return np.dot(th.T, X) + th0


    #returns sigmoid(z). (1xb) numpy array
    def lr_forward(self, th, th0, X):
        return self.sigmoid(self.weighted_sum(th, th0, X))

    def lr_predict_label(self, X):
        return np.where(self. sigmoid(self.weighted_sum(self.th, self.th0, X))>.5, 1, 0)

    #prediction (1 X b) numpy array
    #actual (1 X b ) numpy array
    def nll(self, prediction, actual):
        data_probability = ((actual*np.log(prediction))+((1-actual)*np.log(1-prediction)))
        loss = -np.mean(data_probability, axis=1)
        return loss


    #Th is the weight of LR model. (m X 1) numpy array
    #loss objective with regularizer
    #objective_loss. lr gradient descent should minimize this
    def lr_svm_obj(self, th, th0, X, Y,  lam = .1):
        return self.nll(self.lr_forward(th, th0, X), Y) + lam * np.linalg.norm(th) ** 2 





    def th_update(self, th, th0, X, Y, lam = .1):
        return np.mean((self.lr_forward(th, th0, X) - Y)*X, axis=1,keepdims=True) + 2*lam*th

    def th0_update(self, th, th0, X, Y):
        return np.mean((self.lr_forward(th, th0, X) - Y), axis=1,keepdims=True)


    # a = np.array([[1, 2, 3]])
    # b = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    # print(np.mean(a*b, axis=1, keepdims=True))


    # th = np.array([[2], [3]])
    # th0 = np.array([[1]])
    # x = np.array([[1, 2, 3], [4, 5, 6]])
    # y = np.array([[1, -1, 1]])


    # print(lr_svm_obj(th, th0, x, y, .1)[0]<15)
    # print(th_update(th, th0, 0.0, x, y))
    # print(th0_update(th, th0, x, y))

    def lr_gradient_descent(self, X, Y, iters=100, lrate=0.005, epsilon=.001, lam=.1):
        D, N = X.shape
        th = np.random.normal(0, 1.0 * D ** (-.5), [D, 1])
        th0 = np.zeros((1, 1))
        for it in range(iters):
            i = np.random.randint(N)
            Xt =  X[:, i:i+1]
            Yt =  Y[:, i:i+1]
            th_old = th.copy()
            th0_old = th0.copy()
            # obj_t_minus_1 = lr_svm_obj(th_old, th0_old, X, Y,  lam)
            th = th_old - lrate*self.th_update(th_old, th0_old, Xt, Yt, lam)
            th0 = th0_old - lrate*self.th0_update(th_old, th0_old, Xt, Yt)
            # obj_t = lr_svm_obj(th, th0, X, Y, lam)
            # if abs(obj_t - obj_t_minus_1) < epsilon:
            #     print('stops at iteration:', it)
            #     break
        return th, th0

    def run_lr(self, X, y, iters=100, lrate=0.005, epsilon=.001, lam=.1):
        self.th, self.th0 = \
            self.lr_gradient_descent(X, y, iters=iters, lrate=lrate, epsilon=epsilon, lam=lam)
    
    def get_accuracy(self, X, y):
        guess = self.lr_predict_label(X)
        acc = np.mean(y == guess)
        return acc


def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, y

def xor():
    X = np.array([[1, 2, 1, 2],
                  [1, 2, 2, 1]])
    y = np.array([[1, 1, 0, 0]])
    return X, y

def xor_more():
    '''
    Return d = 2 by n = 4 data matrix and 1 x n = 4 label matrix
    '''
    X = np.array([[1, 2, 1, 2, 2, 4, 1, 3],
                  [1, 2, 2, 1, 3, 1, 3, 3]])
    y = np.array([[1, 1, 0, 0, 1, 1, 0, 0]])
    return X, y

import random
def my_data():
    X = [[], []]
    y = [[]]
    for i in range(50):
        X[0].append(random.randint(0, 40))
        X[1].append(random.randint(0, 100))
        y[0].append(1)
    
    # for i in range(50):
    #     X[0].append(random.randint(40, 60))
    #     X[1].append(random.randint(0, 100))
    #     y[0].append(random.choice([1, 0]))

    for i in range(50):
        X[0].append(random.randint(60, 100))
        X[1].append(random.randint(0, 100))
        y[0].append(0) 

    return np.array(X), np.array(y)
# X, y = xor()

from sklearn.datasets import make_moons


def data2():
    X, y = make_moons(n_samples=100, noise=0.24)

    return X.T, y.reshape(1, y.shape[0])

# th, th0 = lr_gradient_descent(X, y, iters=100000, lrate=0.05, epsilon=.000001, lam=.1)

# data, labels = super_simple_separable()
# data, labels = xor_more()
# data, labels = my_data()
data, labels = data2()

lr = LogisticRegression()
lr.run_lr(data, labels, iters=100000, lrate=.005, epsilon=1e-10, lam=.1)
th, th_0 = lr.th, lr.th0

def plot_lr(X, y, lr):
    th, th0 = lr.th, lr.th0
    ax = plot_data(X, np.where(y!=1, -1, 1))
    plot_separator(ax, th, th0)
    #plt.ioff()
    plt.show()

plot_lr(data, labels, lr)
print(th, th_0)

# def get_accuracy(actual, guess):
#     acc = np.mean(actual == guess)
#     return acc
guess = lr.lr_predict_label(data)
print(lr.get_accuracy(data, labels))
print(lr.lr_predict_label(np.array([[2], [1]])))