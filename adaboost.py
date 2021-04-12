import sys
sys.path.append("/Users/minhqpham/Documents/CWRU/Spring 2021/CSDS 435/libsvm-weights/python")
print(sys.path)

from svmutil import *
from commonutil import *

class classifier:
    def __init__(self, svm, weight=0):
        self.svm = svm
        self.weight = weight
    
    def get_predicted_label(self, y, x):
        p_labels, p_acc, p_vals = svm_predict(y, x, self.svm)
        return p_labels

class boosting:
    def __init__(self):
        self.classifiers = []

    def predict(self, y, x):