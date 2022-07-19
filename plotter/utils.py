import base64
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from django.core.files.images import ImageFile

def plot_line_graph(form_data):
    plt.switch_backend('AGG')
    plt.figure(facecolor="white")

    no_of_scatters = int(form_data.cleaned_data['number_of_lines'])
    params = {}
    #reading the points
    for i in range(1, no_of_scatters+1):
        params['x'+str(i)] = form_data.cleaned_data['x'+str(i)].split(", ")
        params['x'+str(i)] = np.array(params['x'+str(i)]).astype(float)
        params['y'+str(i)] = form_data.cleaned_data['y'+str(i)].split(", ")
        params['y'+str(i)] = np.array(params['y'+str(i)]).astype(float)

        params['data'+str(i)+'_label'] = form_data.cleaned_data['data'+str(i)+'_label']


    #set the scale of the axes
    plt.xscale(form_data.cleaned_data['x_scale'])
    plt.yscale(form_data.cleaned_data['y_scale'])

    plt.title(form_data.cleaned_data['graph_title'])

    plt.xlabel(form_data.cleaned_data['x_axis_label'])
    plt.ylabel(form_data.cleaned_data['y_axis_label'])
    for i in range(1, no_of_scatters+1):
        plt.plot(params['x'+str(i)], params['y'+str(i)], 
                marker='o', label = params['data'+str(i)+'_label'])

    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    image_png = buffer.getvalue()
    graph =  base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph



def plot_bar_graph(form_data):
    plt.switch_backend('AGG')
    plt.figure(facecolor="white")

    no_of_bars = int(form_data.cleaned_data['number_of_bars'])
    params = {}
    #reading the points
    params['data_labels'] = form_data.cleaned_data['data_labels'].split(', ')
    data_labels_len = len(params['data_labels'])
    print(params['data_labels'])
    width = .40

    x = [0]
    for i in range(1, data_labels_len):
        x.append(x[-1]+no_of_bars*width+width)
    x = np.array(x)

    # x = np.arange(data_labels_len)
    # x = np.arange(stop=data_labels_len*2, step=2) #step=2
    print(x)

    for i in range(1, no_of_bars+1):
        params['bar'+str(i)+'_data_values'] = \
            form_data.cleaned_data['bar'+str(i)+'_data_values'].split(", ")
        params['bar'+str(i)+'_data_values'] = \
            np.array(params['bar'+str(i)+'_data_values']).astype(float)
        params['bar'+str(i)+'_label'] = form_data.cleaned_data['bar'+str(i)+'_label']


    #set the scale of the axes
    plt.yscale(form_data.cleaned_data['y_scale'])

    plt.title(form_data.cleaned_data['graph_title'])

    plt.xlabel(form_data.cleaned_data['x_axis_label'])
    plt.ylabel(form_data.cleaned_data['y_axis_label'])
    # if form_data.cleaned_data['oreantation'] == 'horizontal':
    #     for i in range(1, no_of_bars+1):
    #         plt.barh(params['bar'+str(i)+'_data_values'], x+width*(i-1),
    #             width = width,
    #             label=params['bar'+str(i)+'_label'])
    # else:
    for i in range(1, no_of_bars+1):
        plt.bar(x+width*(i-1), params['bar'+str(i)+'_data_values'],
            width = width,
            label=params['bar'+str(i)+'_label'])

    # plt.xticks(x , params['data_labels'])
    plt.xticks(x + ((width*no_of_bars)/2) - width/2 ,params['data_labels'])


    
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    image_png = buffer.getvalue()
    graph =  base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph



def plot_scatter_graph(form_data):
    plt.switch_backend('AGG')
    plt.figure(facecolor="white")

    no_of_scatters = int(form_data.cleaned_data['number_of_scatters'])
    params = {}
    #reading the points
    for i in range(1, no_of_scatters+1):
        params['x'+str(i)] = form_data.cleaned_data['x'+str(i)].split(", ")
        params['x'+str(i)] = np.array(params['x'+str(i)]).astype(float)
        params['y'+str(i)] = form_data.cleaned_data['y'+str(i)].split(", ")
        params['y'+str(i)] = np.array(params['y'+str(i)]).astype(float)

        params['data'+str(i)+'_label'] = form_data.cleaned_data['data'+str(i)+'_label']


    #set the scale of the axes
    plt.xscale(form_data.cleaned_data['x_scale'])
    plt.yscale(form_data.cleaned_data['y_scale'])

    plt.title(form_data.cleaned_data['graph_title'])

    plt.xlabel(form_data.cleaned_data['x_axis_label'])
    plt.ylabel(form_data.cleaned_data['y_axis_label'])
    m = ['.', 'o', 'v', 'p', 's', '*' ]
    for i in range(1, no_of_scatters+1):
        plt.scatter(params['x'+str(i)], params['y'+str(i)], 
                marker=m[i], label = params['data'+str(i)+'_label'])

    #plt.grid()
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    image_png = buffer.getvalue()
    graph =  base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph    


def plot_pie_graph(form_data):
    plt.switch_backend('AGG')
    plt.figure(facecolor="white")

    params = {}
    params['data_labels'] = form_data.cleaned_data['data_labels'].split(", ")
    params['data_values'] = form_data.cleaned_data['data_values'].split(", ")
    params['data_values'] = np.array(params['data_values']).astype(float)
    params['data_values'] = (params['data_values']*100)/np.sum(params['data_values'])



    plt.title(form_data.cleaned_data['graph_title'])

    plt.pie(params['data_values'], 
            labels = params['data_labels'],
            shadow=False,
            autopct='%1.1f%%',
            startangle=90)


    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    image_png = buffer.getvalue()
    graph =  base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph