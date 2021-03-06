from django.urls import path

from . import views

urlpatterns = [
    path("plotter_index/", views.plotter_index, name="plotter_index"),
    path("draw_line_graph/", views.draw_line_graph, name="draw_line_graph"),
    path("draw_bar_graph/", views.draw_bar_graph, name="draw_bar_graph"),
    path("draw_scatter_graph/", views.draw_scatter_graph, name="draw_scatter_graph"),
    path("draw_pie_graph/", views.draw_pie_graph, name="draw_pie_graph"),
    path('error/<str:error_from>/', views.error,name='error'),
]