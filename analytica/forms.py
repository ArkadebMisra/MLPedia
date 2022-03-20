from django import forms
from . models import NeuralNet, Regression

class NuralNetCreateForm(forms.ModelForm):
    class Meta:
        model = NeuralNet
        fields = ('title','model_description', 'features_input', 'labels_input')

class NeuralNetPredictionForm(forms.Form):
    new_data_point = forms.CharField(label='features')

#forms for regression
class RegressionCreateForm(forms.ModelForm):
    class Meta:
        model = Regression
        fields = ('title','degree', 'features_input', 'labels_input')

class RegressionPredictionForm(forms.Form):
    new_data_point = forms.CharField(label='features')