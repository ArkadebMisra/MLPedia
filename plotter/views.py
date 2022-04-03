from django.shortcuts import render

# Create your views here.

from . forms import LineGraphCreateForm



def draw_line_graph(request):
    if request.method == 'POST':
            form = LineGraphCreateForm(request.POST)
            if form.is_valid():
                print('valid form')

    else:
        form = LineGraphCreateForm(data = request.GET)
    return render(request, "plotter/draw_line_graph.html",{
        'form': form,
        'section': 'plotter',
    })
