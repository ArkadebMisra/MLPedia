import numpy as np
import csv
from django.core.files import File
from django.core.files.base import ContentFile

from . neural_net_utils import *

#####################################################
#####read and write NN weights, features, labels #######################

#read features from file for neural net, regression
def read_features(file):
    csvreader = csv.reader(file)
    csvreader.__next__()
    features = []
    for row in csvreader:
        features.append(row)
    features = np.array(features).astype(float).T
    return features


def for_softmax(label, num_of_class):
    softmax_label = [0 for i in range(num_of_class)]
    softmax_label[label] = 1
    return softmax_label

def read_labels(file):
    csvreader = csv.reader(file)
    no_of_class = int(csvreader.__next__()[1])
    labels = []
    for row in csvreader:
        labels.append(for_softmax(int(row[0]), no_of_class))
    labels = np.array(labels).astype(float).T
    return labels

def write_nn(file, nn):
    csvwriter = csv.writer(file)
    for layer in nn.modules:
        if type(layer)==Linear:
            for r in layer.W:
                csvwriter.writerow(r.tolist())
            for r in layer.W0.T:
                csvwriter.writerow(r.tolist())
    return ContentFile(file.getvalue().encode('utf-8'))

def read_saved_nn(filename, read_nn):
    with open(filename, 'r')as file:
        csvreader = csv.reader(file)
        for layer in read_nn.modules:
            if type(layer)==Linear:
                w = []
                w0 = []
                for i in range(layer.m):
                    w_row = csvreader.__next__()
                    w.append(w_row)
                w0.append(csvreader.__next__())
                w = np.array(w).astype(float)
                w0 = np.array(w0).astype(float).T
                layer.W = w
                layer.W0 = w0


# read and write weights/labels for regression model
def read_labels_rg(file):
    csvreader = csv.reader(file)
    _ = csvreader.__next__()
    labels = [[]]
    for row in csvreader:
        labels[0].append(row[0])
    labels = np.array(labels).astype(float)
    return labels

def write_rg(file, rg):
    csvwriter = csv.writer(file)
    csvwriter.writerow(rg.th.T[0].tolist())
    csvwriter.writerow(rg.th0[0].tolist())
    return ContentFile(file.getvalue().encode('utf-8'))

def read_saved_rg(filename, rg):
    with open(filename, 'r')as file:
        csvreader = csv.reader(file)
        th = []
        th0 = []
        th.append(csvreader.__next__())
        th0.append(csvreader.__next__())
        th = np.array(th).astype(float).T
        th0 = np.array(th0).astype(float)
        rg.th = th
        rg.th0 = th0

# read and write weights/labels &
# cluster output for k_means clusturing model

def write_k_means(file, km):
    csvwriter = csv.writer(file)
    for row in km.centroids:
        csvwriter.writerow(row.tolist())
    return ContentFile(file.getvalue().encode('utf-8'))

def read_k_means(filename, km):
    with open(filename, 'r')as file:
        csvreader = csv.reader(file)
        c = []
        for row in csvreader:
            c.append(row)
        c = np.array(c).astype(float)
        km.centroids = c

def write_k_means_output(file, X, y):
    out = np.append(X.T, y.T, axis=1)
    csvwriter = csv.writer(file)
    for row in out:
        csvwriter.writerow(row.tolist())
    return ContentFile(file.getvalue().encode('utf-8'))

