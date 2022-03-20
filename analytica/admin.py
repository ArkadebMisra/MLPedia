from django.contrib import admin

# Register your models here.

from . models import NeuralNet, Regression


admin.site.register(NeuralNet)
admin.site.register(Regression)