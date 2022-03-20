from django.db import models
from django.conf import settings
from django.utils.text import slugify
from django.urls import reverse


# Create your models here.

#model for Neural Network
class NeuralNet(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                            related_name='neural_net_models_created',
                            on_delete=models.CASCADE)
    title = models.CharField(max_length=500)
    model_description = models.CharField(max_length = 10000)
    model_accuracy = models.FloatField(default=0.0)
    features_input = models.FileField(upload_to='files/%Y/%m/%d/')
    labels_input = models.FileField(upload_to='files/%Y/%m/%d/')
    plot = models.ImageField(upload_to='images/%Y/%m/%d/', blank = True)
    weights_file = models.FileField(upload_to='files/%Y/%m/%d/', blank = True)
    slug = models.SlugField(max_length = 600, blank=True)
    created = models.DateField(auto_now_add = True, db_index = True)

    class Meta:
        ordering = ('-created',)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)
        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse('analytica:neural_net_detail', args=[self.id, self.slug])


class Regression(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                            related_name='regression_models_created',
                            on_delete=models.CASCADE)
    title = models.CharField(max_length=500)
    weight_dimention = models.IntegerField(default=0)
    degree = models.IntegerField(default=1)
    model_accuracy = models.FloatField(default=0.0)
    features_input = models.FileField(upload_to='files/%Y/%m/%d/')
    labels_input = models.FileField(upload_to='files/%Y/%m/%d/')
    plot = models.ImageField(upload_to='images/%Y/%m/%d/', blank = True)
    weights_file = models.FileField(upload_to='files/%Y/%m/%d/', blank = True)
    slug = models.SlugField(max_length = 600, blank=True)
    created = models.DateField(auto_now_add = True, db_index = True)

    class Meta:
        ordering = ('-created',)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)
        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse('analytica:regression_detail', args=[self.id, self.slug])