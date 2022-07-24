import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from django.core.files.images import ImageFile


def classify(X, Y, nn):
    plt.switch_backend('AGG')
    D = X.shape[0]
    N = X.shape[1]
    O = Y.shape[0]

    # Draw it...
    def predict(x):
        return nn.modules[-1].class_fun(nn.forward(x))[0]

    xmin, ymin = np.min(X, axis=1) - 1
    xmax, ymax = np.max(X, axis=1) + 1
    print(xmin, ymin, xmax, ymax)
    nax = plot_objective_2d(lambda x: predict(x), xmin, xmax, ymin, ymax)
    plot_data(X, Y, nax)
    buffer = BytesIO()
    plt.savefig(buffer, format="png")

    image_png = buffer
    graph =  ImageFile(image_png)
    return graph


####################
# SUPPORT AND DISPLAY CODE
####################

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])


def cv(value_list):
    return np.transpose(rv(value_list))


def tidy_plot(xmin, xmax, ymin, ymax, center=False, title=None,
              xlabel=None, ylabel=None):
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
    plt.xlim(xmin - eps, xmax + eps)
    plt.ylim(ymin - eps, ymax + eps)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax


def plot_points(x, y, ax=None, clear=False,
                xmin=None, xmax=None, ymin=None, ymax=None,
                style='or-', equal=False):
    padup = lambda v: v + 0.05 * abs(v)
    paddown = lambda v: v - 0.05 * abs(v)
    if ax is None:
        if xmin == None: xmin = paddown(np.min(x))
        if xmax == None: xmax = padup(np.max(x))
        if ymin == None: ymin = paddown(np.min(y))
        if ymax == None: ymax = padup(np.max(y))
        ax = tidy_plot(xmin, xmax, ymin, ymax)
        x_range = xmax - xmin;
        y_range = ymax - ymin
        if equal and .1 < x_range / y_range < 10:
            # ax.set_aspect('equal')
            plt.axis('equal')
            if x_range > y_range:
                ax.set_xlim((xmin, xmax))
            else:
                ax.set_ylim((ymin, ymax))
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    elif clear:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x, y, style, markeredgewidth=0.0, linewidth=5.0)
    # Seems to occasionally mess up the limits
    ax.set_xlim(xlim);
    ax.set_ylim(ylim)
    ax.grid(True, which='both')
    return ax


def add_ones(X):
    return np.vstack([X, np.ones(X.shape[1])])


def plot_data(data, labels, ax=None,
              xmin=None, xmax=None, ymin=None, ymax=None):
    # Handle 1D data
    if data.shape[0] == 1:
        data = add_ones(data)
    if ax is None:
        if xmin == None: xmin = np.min(data[0, :]) - 0.5
        if xmax == None: xmax = np.max(data[0, :]) + 0.5
        if ymin == None: ymin = np.min(data[1, :]) - 0.5
        if ymax == None: ymax = np.max(data[1, :]) + 0.5
        ax = tidy_plot(xmin, xmax, ymin, ymax)

        x_range = xmax - xmin
        y_range = ymax - ymin
        if .1 < x_range / y_range < 10:
            ax.set_aspect('equal')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for yi in set([int(_y) for _y in set(labels.flatten().tolist())]):
        color = ['r', 'g', 'b'][yi]
        marker = ['X', 'o', 'v'][yi]
        cl = np.where(labels[1, :] == yi)
        ax.scatter(data[0, cl], data[1, cl], c=color, marker=marker, s=50,
                   edgecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, which='both')
    return ax


def plot_data_linear(data, labels, ax = None, clear = False,
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



def plot_objective_2d(J, xmin=-5, xmax=5,
                      ymin=-5, ymax=5,
                      cmin=None, cmax=None,
                      res=50, ax=None):
    if ax is None:
        ax = tidy_plot(xmin, xmax, ymin, ymax)
    else:
        if xmin == None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

    ima = np.array([[J(cv([x1i, x2i])) \
                     for x1i in np.linspace(xmin, xmax, res)] \
                    for x2i in np.linspace(ymin, ymax, res)])
    im = ax.imshow(np.flipud(ima), interpolation='none',
                   extent=[xmin, xmax, ymin, ymax],
                   cmap='viridis')
    if cmin is not None or cmax is not None:
        if cmin is None: cmin = min(ima)
        if cmax is None: cmax = max(ima)
        im.set_clim(cmin, cmax)
    plt.colorbar(im)
    return ax

def plot_nn(X, Y, nn):
    """ Plot output of nn vs. data """
    plt.switch_backend('AGG')
    def predict(x):
        return nn.modules[-1].class_fun(nn.forward(x))[0]
    xmin, ymin = np.min(X, axis=1)-1
    xmax, ymax = np.max(X, axis=1)+1
    nax = plot_objective_2d(lambda x: predict(x), xmin, xmax, ymin, ymax)
    plot_data(X, Y, nax)
    buffer = BytesIO()
    plt.savefig(buffer, format="png")

    image_png = buffer
    graph =  ImageFile(image_png)
    return graph


#plotting regression model for 2D data

def plot_regression_model(X, y, rg):
    """ Plot output of regression vs. data """
    plt.switch_backend('AGG')
    plt.figure(facecolor="white")
    ax = plt.subplot()
    ax.scatter(X, y, 2, c='#F2668B')
    
    min_X = np.min(X)
    max_X = np.max(X)

    no_of_rg_line_points = 10
    
    rg_line_X = np.linspace(min_X, max_X, no_of_rg_line_points)

    # rg_line_X = np.arange(min_X, max_X, (max_X-min_X)/no_of_rg_line_points)
    # rg_line_X = rg_line_X.reshape((1, rg_line_X.shape[0]))
    # guess = rg.predict(X)
    # ax.scatter(X, guess , 2, '#FF0000')

    guess = rg.predict(rg_line_X.reshape(1, no_of_rg_line_points))
    ax.plot(rg_line_X, guess.reshape((10,)), c= "#032642", marker='.', linestyle=':')
    # print(rg_line_X.shape)
    # print(guess.shape)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")

    image_png = buffer
    graph =  ImageFile(image_png)
    return graph

#plotting k means cluster model for 2D data

def plot_k_means_model(X, km):

    plt.switch_backend('AGG')
    plt.figure(facecolor="white")
    plt.style.use('seaborn')
    ax = plt.subplot()

    c, y = km.centroids, km.assigned_labels
    ax.scatter(X[0:1, :], X[1:2, :], 20, y[0:1, :], cmap='Spectral', edgecolor='k')
    n = [i for i in range(km.k)]
    for i, txt in enumerate(n):
        ax.annotate(txt, (c[0:1, i:i+1], c[1:2, i:i+1]), fontsize=25)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")

    image_png = buffer
    graph =  ImageFile(image_png)
    return graph


###########plot logistic regression

def plot_lr(X, y, lr):

    plt.switch_backend('AGG')
    plt.figure(facecolor="white")

    th, th0 = lr.th, lr.th0
    ax = plot_data_linear(X, np.where(y!=1, -1, 1))
    plot_separator(ax, th, th0)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")

    image_png = buffer
    graph =  ImageFile(image_png)
    return graph


def plot_perceptron(X, y, pc):
    plt.switch_backend('AGG')
    plt.figure(facecolor="white")

    th, th_0 = pc.th, pc.th0
    ax = plot_data_linear(X, y)
    plot_separator(ax, th, th_0)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")

    image_png = buffer
    graph =  ImageFile(image_png)
    return graph