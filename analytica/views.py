from django.shortcuts import render, redirect, get_object_or_404
from django import forms
from django.contrib.auth.decorators import login_required


import io
import numpy as np



from . models import NeuralNet, Regression
from . forms import NuralNetCreateForm, NeuralNetPredictionForm, \
                    RegressionCreateForm, RegressionPredictionForm

from . import neural_net_utils as nn_utils
from . import regression_utils as rg_utils
from . disp_utils import *
from . import file_manager as fm

# Create your views here.

#views for neural network

@login_required
def neural_net_create(request):
    if request.method == 'POST':
        form = NuralNetCreateForm(request.POST, request.FILES)
        if form.is_valid():
            new_item =form.save(commit=False)
            new_item.user = request.user

            uploaded_features = request.FILES['features_input']
            uploaded_features = uploaded_features.read().decode('UTF-8')
            feature_buffer = io.StringIO(uploaded_features)

            X = fm.read_features(feature_buffer)
            feature_dimention = X.shape[0]


            uploaded_labels = request.FILES['labels_input']
            uploaded_labels = uploaded_labels.read().decode('UTF-8')
            labels_buffer = io.StringIO(uploaded_labels)

            Y = fm.read_labels(labels_buffer)

            nn = nn_utils.create_nn_from_description(
                        nn_utils.process_model_description(
                        str(new_item.model_description)))
            #nn = Sequential([Linear(2, 4), ReLU(), Linear(4, 2), ReLU(), Linear(2,2), SoftMax()], NLL())

            # Modifies the weights and biases
            nn.sgd(X, Y, iters=100000, lrate=0.005)

            weight_buffer = io.StringIO()
            weights_file = fm.write_nn(weight_buffer, nn)

            new_item.model_accuracy = nn.get_accuracy(X, Y)
            if feature_dimention <=2:
                new_item.plot.save(new_item.title+".png",plot_nn(X, Y, nn))

            new_item.weights_file.save(new_item.title+"_weights.csv", weights_file)
            # write_nn(new_item.weights_file.path, nn)
            new_item.save()

    else:
        form = NuralNetCreateForm(data = request.GET)
    return render(request, "analytica/neural_nets/create_neural_net.html",{
        'form': form,
        'section': 'analytica',
    })


@login_required
def neural_net_models_list(request):
    models = NeuralNet.objects.filter(user=request.user).order_by('-created')
    return render(request, "analytica/neural_nets/neural_net_models_list.html",{
        'models': models,
        'section': 'analytica',
    })


@login_required
def neural_net_detail(request, model_id, model):
    nn_model = get_object_or_404(NeuralNet,id = model_id, slug=model)
    if request.method == 'POST':
        submitted_form = NeuralNetPredictionForm(request.POST)
        if submitted_form.is_valid():
            new_data_point = submitted_form.cleaned_data['new_data_point']
            new_data_point = [str(new_data_point).split(',')]
            new_data_point = np.array(new_data_point).astype(float).T
            nn = nn_utils.create_nn_from_description(
                        nn_utils.process_model_description(
                        str(nn_model.model_description)))
            fm.read_saved_nn(nn_model.weights_file.path, nn)
            prediction = nn.predict(new_data_point)
            acc = 0#nn_model['model_accuracy']
        return render(request,'analytica/neural_nets/neural_net_detail.html', 
                        {'nn_model': nn_model, 
                        'form': submitted_form, 
                        'prediction': prediction})
    else:
        prediction = None
        return render(request,'analytica/neural_nets/neural_net_detail.html', 
                        {'nn_model': nn_model, 
                        'form': NeuralNetPredictionForm(), 
                        'prediction': prediction})  


#views for linear and polynomial regression

@login_required
def regression_create(request):
    if request.method == 'POST':
        form = RegressionCreateForm(request.POST, request.FILES)
        if form.is_valid():
            new_item =form.save(commit=False)
            new_item.user = request.user

            uploaded_features = request.FILES['features_input']
            uploaded_features = uploaded_features.read().decode('UTF-8')
            feature_buffer = io.StringIO(uploaded_features)

            X = fm.read_features(feature_buffer)
            feature_dimention = X.shape[0]
            new_item.weight_dimention = feature_dimention


            uploaded_labels = request.FILES['labels_input']
            uploaded_labels = uploaded_labels.read().decode('UTF-8')
            labels_buffer = io.StringIO(uploaded_labels)

            y = fm.read_labels_rg(labels_buffer)
            rg = rg_utils.RegressionModel(X.shape[0], new_item.degree)

            # Modifies the weights and biases
            rg.run_regression(X, y, iters=10000, lrate=0.005, lam=0)

            weight_buffer = io.StringIO()
            weights_file = fm.write_rg(weight_buffer, rg)

            new_item.model_accuracy = rg.r2_score(X, y)
            if feature_dimention <=1 and new_item.degree==1:
                new_item.plot.save(new_item.title+".png",plot_regression_model(X, y, rg))

            new_item.weights_file.save(new_item.title+"_weights.csv", weights_file)
            # write_nn(new_item.weights_file.path, nn)
            new_item.save()

    else:
        form = RegressionCreateForm(data = request.GET)
    return render(request, "analytica/regression/create_regression.html",{
        'form': form,
        'section': 'analytica',
    })


@login_required
def regression_models_list(request):
    models = Regression.objects.filter(user=request.user).order_by('-created')
    return render(request, "analytica/regression/regression_models_list.html",{
        'models': models,
        'section': 'analytica',
    })


@login_required
def regression_detail(request, model_id, model):
    rg_model = get_object_or_404(Regression,id = model_id, slug=model)
    if request.method == 'POST':
        submitted_form = RegressionPredictionForm(request.POST)
        if submitted_form.is_valid():
            new_data_point = submitted_form.cleaned_data['new_data_point']
            new_data_point = [str(new_data_point).split(',')]
            new_data_point = np.array(new_data_point).astype(float).T
            rg = rg_utils.RegressionModel(rg_model.weight_dimention, rg_model.degree)
            fm.read_saved_rg(rg_model.weights_file.path, rg)
            prediction = rg.predict(new_data_point)
        return render(request,'analytica/regression/regression_detail.html', 
                        {'rg_model': rg_model, 
                        'form': submitted_form, 
                        'prediction': prediction})
    else:
        prediction = None
        return render(request,'analytica/regression/regression_detail.html', 
                        {'rg_model': rg_model, 
                        'form': RegressionPredictionForm(), 
                        'prediction': prediction})  
