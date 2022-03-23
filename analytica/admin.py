from django.contrib import admin

# Register your models here.

from . models import NeuralNet, Regression, KMeansCluster, \
                    LogisticRegression, Perceptron


admin.site.register(NeuralNet)
admin.site.register(Regression)
admin.site.register(KMeansCluster)
admin.site.register(LogisticRegression)
admin.site.register(Perceptron)