from django.urls import path

from . import views

urlpatterns = [
    #urls for Neural Net
    path("nn_create/",views.neural_net_create, name = "neural_net_create"),
    path("nn_models_list/",views.neural_net_models_list, name = "neural_net_models_list"),
    path('nn_detail/<int:model_id>/<slug:model>/', 
        views.neural_net_detail,name='neural_net_detail'),

    #urls for regression
    path("rg_create/",views.regression_create, name = "regression_create"),
    path("rg_models_list/",views.regression_models_list, name = "regression_models_list"),
    path('rg_detail/<int:model_id>/<slug:model>/', 
        views.regression_detail,name='regression_detail'),
]