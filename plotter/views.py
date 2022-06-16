from django.shortcuts import render
from django.urls import reverse
from django.http import HttpResponse,HttpResponseRedirect, Http404
from django.contrib import messages

# Create your views here.

from . forms import LineGraphCreateForm, BarGraphCreateForm, \
    ScatterGraphCreateForm, PieGraphCreateForm

from . utils import plot_bar_graph, plot_line_graph, plot_pie_graph, plot_scatter_graph


def plotter_index(request):
    return render(request, "plotter/plotter_index.html", {})


def error(request, error_from):
    return render(request, "plotter/error.html",{'error_from':error_from})

def draw_line_graph(request):
    try:
        plot = None
        if request.method == 'POST':
                form = LineGraphCreateForm(request.POST)
                if form.is_valid():
                    plot = plot_line_graph(form)

        else:
            form = LineGraphCreateForm(data = request.GET)
    except:
        messages.error(request, "please enter the data in correct format")
    return render(request, "plotter/draw_line_graph.html",{
        'form': form,
        'plot': plot,
        'section': 'plotter',
    })

def draw_bar_graph(request):
    try:
        plot = None
        if request.method == 'POST':
                form = BarGraphCreateForm(request.POST)
                if form.is_valid():
                    plot = plot_bar_graph(form)

        else:
            form = BarGraphCreateForm(data = request.GET)
    except:
        messages.error(request, "please enter the data in correct format")
    return render(request, "plotter/draw_bar_graph.html",{
        'form': form,
        'plot': plot,
        'section': 'plotter',
    })

def draw_scatter_graph(request):
    try:
        plot = None
        if request.method == 'POST':
                form = ScatterGraphCreateForm(request.POST)
                if form.is_valid():
                    plot = plot_scatter_graph(form)

        else:
            form = ScatterGraphCreateForm(data = request.GET)
    except:
        messages.error(request, "please enter the data in correct format")
    return render(request, "plotter/draw_scatter_graph.html",{
        'form': form,
        'plot': plot,
        'section': 'plotter',
    })

def draw_pie_graph(request):
    try:
        plot = None
        if request.method == 'POST':
                form = PieGraphCreateForm(request.POST)
                if form.is_valid():
                    plot = plot_pie_graph(form)

        else:
            form = PieGraphCreateForm(data = request.GET)
    except:
        messages.error(request, "please enter the data in correct format")
    return render(request, "plotter/draw_pie_graph.html",{
        'form': form,
        'plot': plot,
        'section': 'plotter',
    })