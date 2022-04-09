from django.shortcuts import render

# Create your views here.

from . forms import LineGraphCreateForm, BarGraphCreateForm, \
    ScatterGraphCreateForm, PieGraphCreateForm

from . utils import plot_bar_graph, plot_line_graph, plot_pie_graph, plot_scatter_graph


def plotter_index(request):
    return render(request, "plotter/plotter_index.html", {})


def draw_line_graph(request):
    plot = None
    if request.method == 'POST':
            form = LineGraphCreateForm(request.POST)
            if form.is_valid():
                plot = plot_line_graph(form)

    else:
        form = LineGraphCreateForm(data = request.GET)
    return render(request, "plotter/draw_line_graph.html",{
        'form': form,
        'plot': plot,
        'section': 'plotter',
    })

def draw_bar_graph(request):
    plot = None
    if request.method == 'POST':
            form = BarGraphCreateForm(request.POST)
            if form.is_valid():
                plot = plot_bar_graph(form)

    else:
        form = BarGraphCreateForm(data = request.GET)
    return render(request, "plotter/draw_bar_graph.html",{
        'form': form,
        'plot': plot,
        'section': 'plotter',
    })

def draw_scatter_graph(request):
    plot = None
    if request.method == 'POST':
            form = ScatterGraphCreateForm(request.POST)
            if form.is_valid():
                plot = plot_scatter_graph(form)

    else:
        form = ScatterGraphCreateForm(data = request.GET)
    return render(request, "plotter/draw_scatter_graph.html",{
        'form': form,
        'plot': plot,
        'section': 'plotter',
    })

def draw_pie_graph(request):
    plot = None
    if request.method == 'POST':
            form = PieGraphCreateForm(request.POST)
            if form.is_valid():
                plot = plot_pie_graph(form)

    else:
        form = PieGraphCreateForm(data = request.GET)
    return render(request, "plotter/draw_pie_graph.html",{
        'form': form,
        'plot': plot,
        'section': 'plotter',
    })