import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def plot_line_graph():
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
