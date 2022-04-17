from django.urls import path

from . import views

urlpatterns = [
    path("analytica_index", views.analytica_index, name="analytica_index"),
    #urls for Neural Net
    path("nn_create/",views.neural_net_create, name = "neural_net_create"),
    path("nn_models_list/",views.neural_net_models_list, name = "neural_net_models_list"),
    path('nn_detail/<int:model_id>/<slug:model>/', 
        views.neural_net_detail,name='neural_net_detail'),
    path('nn_delete/<int:model_id>/<slug:model>/', 
        views.neural_net_delete,name='neural_net_delete'),

    #urls for regression
    path("rg_create/",views.regression_create, name = "regression_create"),
    path("rg_models_list/",views.regression_models_list, name = "regression_models_list"),
    path('rg_detail/<int:model_id>/<slug:model>/', 
        views.regression_detail,name='regression_detail'),
    path('rg_delete/<int:model_id>/<slug:model>/', 
        views.regression_delete,name='regression_delete'),


    #urls for k_means
    path("km_create/",views.k_means_create, name = "k_means_create"),
    path("km_models_list/",views.k_means_models_list, name = "k_means_models_list"),
    path('km_detail/<int:model_id>/<slug:model>/', 
        views.k_means_detail,name='k_means_detail'),
    path('km_delete/<int:model_id>/<slug:model>/', 
        views.k_means_delete,name='k_means_delete'),

    #urls for logistic regression
    path("lr_create/",views.logistic_regression_create, 
        name = "logistic_regression_create"),
    path("lr_models_list/",views.logistic_regression_models_list, 
        name = "logistic_regression_models_list"),
    path('lr_detail/<int:model_id>/<slug:model>/', 
        views.logistic_regression_detail,name='logistic_regression_detail'),
    path('lr_delete/<int:model_id>/<slug:model>/', 
        views.logistic_regression_delete,name='logistic_regression_delete'),


    #urls for Perceptron
    path("pc_create/",views.perceptron_create, name = "perceptron_create"),
    path("pc_models_list/",views.perceptron_models_list, name = "perceptron_models_list"),
    path('pc_detail/<int:model_id>/<slug:model>/', 
        views.perceptron_detail,name='perceptron_detail'),
    path('pc_delete/<int:model_id>/<slug:model>/', 
        views.perceptron_delete,name='perceptron_delete'),
]