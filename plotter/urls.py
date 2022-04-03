from django.urls import path

from . import views

urlpatterns = [
    path("draw_line_graph", views.draw_line_graph, name="draw_line_graph"),
]