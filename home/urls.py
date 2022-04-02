from django.urls import path

from . import views

urlpatterns = [
    #urls for Neural Net
    path("",views.home, name = "home"),
]