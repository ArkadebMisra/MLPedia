from django.shortcuts import render, redirect, get_object_or_404
from django import forms
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse,HttpResponseRedirect, Http404
from django.contrib import messages

import io
import numpy as np



from . models import NeuralNet, Regression,  KMeansCluster, \
                        LogisticRegression, Perceptron
from . forms import NuralNetCreateForm, NeuralNetPredictionForm, \
                    RegressionCreateForm, RegressionPredictionForm, \
                    KMeansCreateForm, KMeansPredictionForm, \
                    LogisticRegressionCreateForm, LogisticRegressionPredictionForm, \
                    PerceptronCreateForm, PerceptronPredictionForm

from . import neural_net_utils as nn_utils
from . import regression_utils as rg_utils
from . import k_means_utils as km_utils
from . import logistic_regression_utils as lr_utils
from . import perceptron_utils as pc_utils
from . disp_utils import *
from . import file_manager as fm

# Create your views here.


def analytica_index(request):
    return render(request, 'analytica/analytica_index.html', {})
    

#views for neural network

@login_required
def neural_net_create(request):
    try:
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
                nn.sgd(X, Y, iters=int(request.POST['iterations']), lrate=0.005)

                weight_buffer = io.StringIO()
                weights_file = fm.write_nn(weight_buffer, nn)

                new_item.model_accuracy = nn.get_accuracy(X, Y)
                if feature_dimention <=2:
                    new_item.plot.save(new_item.title+".png",plot_nn(X, Y, nn))

                new_item.weights_file.save(new_item.title+"_weights.csv", weights_file)
                # write_nn(new_item.weights_file.path, nn)
                new_item.save()
                messages.success(request, 'Model added successfully')
                return HttpResponseRedirect(reverse('analytica:neural_net_models_list'))

        else:
            form = NuralNetCreateForm(data = request.GET)
    except:
        messages.error(request, "oops!something went wrong <br>please enter the data in correct format")
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
    prediction = None
    nn_model = get_object_or_404(NeuralNet,id = model_id, slug=model)
    try:
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
        else:
            submitted_form = NeuralNetPredictionForm(request.GET)
    except:
        # messages.error(request, "please enter the data in correct format")
        messages.error(request, "oops!something went wrong <br>please enter the data in correct format")
    return render(request,'analytica/neural_nets/neural_net_detail.html', 
                    {'nn_model': nn_model, 
                    'form': submitted_form, 
                    'prediction': prediction})  


@login_required
def neural_net_delete(request,model_id,model):
    nn_model = get_object_or_404(NeuralNet,id = model_id, slug=model)
    if nn_model.user != request.user:
        raise Http404
    else:
        if request.method == 'POST':
            nn_model.delete()
            return HttpResponseRedirect(reverse('analytica:neural_net_models_list'))

    return render(request,
                    'analytica/neural_nets/neural_net_delete.html',
                    {'nn_model':nn_model})

                    
#views for linear and polynomial regression

@login_required
def regression_create(request):
    try:
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
                rg.run_regression(X, y, iters=int(request.POST['iterations']), lrate=0.005, lam=0)

                weight_buffer = io.StringIO()
                weights_file = fm.write_rg(weight_buffer, rg)

                new_item.model_accuracy = rg.r2_score(X, y)
                if feature_dimention <=1:
                    new_item.plot.save(new_item.title+".png",plot_regression_model(X, y, rg))

                new_item.weights_file.save(new_item.title+"_weights.csv", weights_file)
                # write_nn(new_item.weights_file.path, nn)
                new_item.save()
                messages.success(request, 'Model added successfully')
                return HttpResponseRedirect(reverse('analytica:regression_models_list'))

        else:
            form = RegressionCreateForm(data = request.GET)
    except:
        # return HttpResponseRedirect(
        #     reverse('plotter:error', kwargs={'error_from':'regression_create'})) 
        messages.error(request, "oops!something went wrong <br>please enter the data in correct format")   
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
    prediction = None
    rg_model = get_object_or_404(Regression,id = model_id, slug=model)
    try:    
        if request.method == 'POST':
            submitted_form = RegressionPredictionForm(request.POST)
            if submitted_form.is_valid():
                new_data_point = submitted_form.cleaned_data['new_data_point']
                new_data_point = [str(new_data_point).split(',')]
                new_data_point = np.array(new_data_point).astype(float).T
                rg = rg_utils.RegressionModel(rg_model.weight_dimention, rg_model.degree)
                fm.read_saved_rg(rg_model.weights_file.path, rg)
                prediction = rg.predict(new_data_point)
        else:
            submitted_form = RegressionPredictionForm(request.GET)
    except:
        # messages.error(request, "please enter the data in correct format")
        messages.error(request, "oops!something went wrong <br>please enter the data in correct format")
    return render(request,'analytica/regression/regression_detail.html', 
                    {'rg_model': rg_model, 
                    'form': submitted_form, 
                    'prediction': prediction})

@login_required
def regression_delete(request,model_id,model):
    rg_model = get_object_or_404(Regression,id = model_id, slug=model)
    if rg_model.user != request.user:
        raise Http404
    else:
        if request.method == 'POST':
            rg_model.delete()
            return HttpResponseRedirect(reverse('analytica:regression_models_list'))

    return render(request,
                    'analytica/regression/regression_delete.html',
                    {'rg_model':rg_model})  


############# Views for K Means Clusturing #########################

@login_required
def k_means_create(request):
    try:
        if request.method == 'POST':
            form = KMeansCreateForm(request.POST, request.FILES)
            if form.is_valid():
                new_item =form.save(commit=False)
                new_item.user = request.user

                uploaded_features = request.FILES['features_input']
                uploaded_features = uploaded_features.read().decode('UTF-8')
                feature_buffer = io.StringIO(uploaded_features)

                X = fm.read_features(feature_buffer)
                feature_dimention = X.shape[0]

                km = km_utils.KMeansClustur(new_item.no_of_clusters)

                # Modifies the weights and biases
                km.run_k_means(X, iter=int(request.POST['iterations']))

                weight_buffer = io.StringIO()
                weights_file = fm.write_k_means(weight_buffer, km)

                output_buffer = io.StringIO()
                output_file = fm.write_k_means_output(output_buffer, X, km.assigned_labels)

                if feature_dimention <=2:
                    new_item.plot.save(new_item.title+".png",plot_k_means_model(X, km))

                new_item.weights_file.save(new_item.title+"_weights.csv", weights_file)
                new_item.k_means_output_file.save(new_item.title+"_cluster_output.csv",
                                                    output_file)
                # write_nn(new_item.weights_file.path, nn)
                new_item.save()
                messages.success(request, 'Model added successfully')
                return HttpResponseRedirect(reverse('analytica:k_means_models_list'))

        else:
            form = KMeansCreateForm(data = request.GET)
    except:
        # return HttpResponseRedirect(
        #     reverse('plotter:error', kwargs={'error_from':'km_create'}))
        messages.error(request, "oops!something went wrong <br>please enter the data in correct format")
    return render(request, "analytica/k_means/create_k_means.html",{
        'form': form,
        'section': 'analytica',
    })


@login_required
def k_means_models_list(request):
    models = KMeansCluster.objects.filter(user=request.user).order_by('-created')
    return render(request, "analytica/k_means/k_means_models_list.html",{
        'models': models,
        'section': 'analytica',
    })


@login_required
def k_means_detail(request, model_id, model):
    prediction = None
    km_model = get_object_or_404(KMeansCluster,id = model_id, slug=model)
    try:
        if request.method == 'POST':
            submitted_form = KMeansPredictionForm(request.POST)
            if submitted_form.is_valid():
                new_data_point = submitted_form.cleaned_data['new_data_point']
                new_data_point = [str(new_data_point).split(',')]
                new_data_point = np.array(new_data_point).astype(float).T
                km = km_utils.KMeansClustur(km_model.no_of_clusters)
                fm.read_k_means(km_model.weights_file.path, km)
                prediction = km.assign_label(new_data_point)
        else:
            submitted_form = KMeansPredictionForm(request.GET)
    except:
        # messages.error(request, "please enter the data in correct format")
        messages.error(request, "oops!something went wrong <br>please enter the data in correct format")
    return render(request,'analytica/k_means/k_means_detail.html', 
                    {'km_model': km_model, 
                    'form': submitted_form, 
                    'prediction': prediction})  



@login_required
def k_means_delete(request,model_id,model):
    km_model = get_object_or_404(KMeansCluster ,id = model_id, slug=model)
    if km_model.user != request.user:
        raise Http404
    else:
        if request.method == 'POST':
            km_model.delete()
            return HttpResponseRedirect(reverse('analytica:k_means_models_list'))

    return render(request,
                    'analytica/k_means/k_means_delete.html',
                    {'km_model':km_model})  


###########Views for Logistic Regression##############################

@login_required
def logistic_regression_create(request):
    try:
        if request.method == 'POST':
            form = LogisticRegressionCreateForm(request.POST, request.FILES)
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

                y = fm.read_labels_rg(labels_buffer)

                lr = lr_utils.LogisticRegression()
                # print(y)

                # Modifies the weights and biases
                lr.run_lr(X, y, iters=int(request.POST['iterations']), lrate=0.005, epsilon=.001, lam=.1)

                weight_buffer = io.StringIO()
                weights_file = fm.write_rg(weight_buffer, lr)

                new_item.model_accuracy = lr.get_accuracy(X, y)
                if feature_dimention <=2:
                    new_item.plot.save(new_item.title+".png",plot_lr(X, y, lr))

                new_item.weights_file.save(new_item.title+"_weights.csv", weights_file)
                # write_nn(new_item.weights_file.path, nn)
                new_item.save()
                messages.success(request, 'Model added successfully')
                return HttpResponseRedirect(reverse('analytica:logistic_regression_models_list'))

        else:
            form = LogisticRegressionCreateForm(data = request.GET)
    except:
        # return HttpResponseRedirect(
        #     reverse('plotter:error', kwargs={'error_from':'lr_create'}))
        messages.error(request, "oops!something went wrong <br>please enter the data in correct format")
    return render(request, "analytica/logistic_regression/create_Logistic_regression.html",{
        'form': form,
        'section': 'analytica',
    })


@login_required
def logistic_regression_models_list(request):
    models = LogisticRegression.objects.filter(user=request.user).order_by('-created')
    return render(request, 
        "analytica/logistic_regression/logistic_regression_models_list.html",{
        'models': models,
        'section': 'analytica',
    })


@login_required
def logistic_regression_detail(request, model_id, model):
    prediction = None
    lr_model = get_object_or_404(LogisticRegression,id = model_id, slug=model)
    try:
        if request.method == 'POST':
            submitted_form = LogisticRegressionPredictionForm(request.POST)
            if submitted_form.is_valid():
                new_data_point = submitted_form.cleaned_data['new_data_point']
                new_data_point = [str(new_data_point).split(',')]
                new_data_point = np.array(new_data_point).astype(float).T
                lr = lr_utils.LogisticRegression()
                fm.read_saved_rg(lr_model.weights_file.path, lr)
                prediction = lr.lr_predict_label(new_data_point)
        else:
            submitted_form = LogisticRegressionPredictionForm(request.GET)
    except:
        # messages.error(request, "please enter the data in correct format")
        messages.error(request, "oops!something went wrong <br>please enter the data in correct format")
    return render(request,
                    "analytica/logistic_regression/logistic_regression_detail.html", 
                    {'lr_model': lr_model, 
                    'form': submitted_form, 
                    'prediction': prediction})

@login_required
def logistic_regression_delete(request,model_id,model):
    lr_model = get_object_or_404(LogisticRegression ,id = model_id, slug=model)
    if lr_model.user != request.user:
        raise Http404
    else:
        if request.method == 'POST':
            lr_model.delete()
            return HttpResponseRedirect(reverse('analytica:logistic_regression_models_list'))

    return render(request,
                    'analytica/logistic_regression/logistic_regression_delete.html',
                    {'lr_model':lr_model})   

###########Views for perceptron##############################

@login_required
def perceptron_create(request):
    try:
        if request.method == 'POST':
            form = PerceptronCreateForm(request.POST, request.FILES)
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

                y = fm.read_labels_rg(labels_buffer)

                pc = pc_utils.Perceptron()

                # Modifies the weights and biases
                pc.run_perceptron(X, y, T=int(request.POST['iterations']))

                weight_buffer = io.StringIO()
                weights_file = fm.write_rg(weight_buffer, pc)

                new_item.model_accuracy = pc.get_accuracy(X, y)
                if feature_dimention <=2:
                    new_item.plot.save(new_item.title+".png",plot_perceptron(X, y, pc))

                new_item.weights_file.save(new_item.title+"_weights.csv", weights_file)
                # write_nn(new_item.weights_file.path, nn)
                new_item.save()
                messages.success(request, 'Model added successfully')
                return HttpResponseRedirect(reverse('analytica:perceptron_models_list'))

        else:
            form = PerceptronCreateForm(data = request.GET)
    except:
        # return HttpResponseRedirect(
        #     reverse('plotter:error', kwargs={'error_from':'perceptron_create'}))
        messages.error(request, "oops!something went wrong <br>please enter the data in correct format")
    return render(request, "analytica/perceptron/create_perceptron.html",{
        'form': form,
        'section': 'analytica',
    })



@login_required
def perceptron_models_list(request):
    models = Perceptron.objects.filter(user=request.user).order_by('-created')
    return render(request, 
        "analytica/perceptron/perceptron_models_list.html",{
        'models': models,
        'section': 'analytica',
    })


@login_required
def perceptron_detail(request, model_id, model):
    pc_model = get_object_or_404(Perceptron,id = model_id, slug=model)
    prediction = None
    try:
        if request.method == 'POST':
            submitted_form = PerceptronPredictionForm(request.POST)
            if submitted_form.is_valid():
                new_data_point = submitted_form.cleaned_data['new_data_point']
                new_data_point = [str(new_data_point).split(',')]
                new_data_point = np.array(new_data_point).astype(float).T
                pc = pc_utils.Perceptron()
                fm.read_saved_rg(pc_model.weights_file.path, pc)
                prediction = pc.predict_label(new_data_point)
        else:
            submitted_form = PerceptronPredictionForm(data = request.GET)
    except:
        # messages.error(request, "please enter the data in correct format")
        messages.error(request, "oops!something went wrong <br>please enter the data in correct format")
    return render(request,
                    "analytica/perceptron/perceptron_detail.html", 
                    {'pc_model': pc_model, 
                    'form': submitted_form, 
                    'prediction': prediction})

@login_required
def perceptron_delete(request,model_id,model):
    pc_model = get_object_or_404(Perceptron ,id = model_id, slug=model)
    if pc_model.user != request.user:
        raise Http404
    else:
        if request.method == 'POST':
            pc_model.delete()
            return HttpResponseRedirect(reverse('analytica:perceptron_models_list'))

    return render(request,
                    'analytica/perceptron/perceptron_delete.html',
                    {'pc_model':pc_model})   