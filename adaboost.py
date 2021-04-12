import sys
sys.path.append("/Users/minhqpham/Documents/CWRU/Spring 2021/CSDS 435/libsvm-weights/python")
print(sys.path)

from svmutil import *
from commonutil import *
import numpy as np

class classifier:
    def __init__(self, svm, weight=0):
        self.svm = svm
        self.weight = weight

    def update_weight(self, err):
        if err == 0:
            self.weight = 10000.0
        self.weight = 0.5 * np.log((1.0-err)/err)
    
    def get_predicted_label(self, y, x):
        p_labels, p_acc, p_vals = svm_predict(y, x, self.svm)
        return p_labels

class classifier_set:
    def __init__(self):
        self.classifiers = []

    def predict(self, y, x):
        predictions = [0] * len(x)
        for c in self.classifiers:
            p_labels = c.get_predicted_label(y, x)
            for i in range(len(x)):
                predictions[i] += (c.weight * p_labels[i])
        
        matches = 0
        for i in range(len(predictions)):
            if predictions[i] >= 0:
                predictions[i] = 1
            else:
                predictions[i] = -1
            if y[i] == predictions[i]:
                matches += 1
        
        accuracy = matches/len(y) * 100
        return predictions, accuracy

    def add(self, c):
        self.classifiers.append(c)

def adaboost(record_weights, y, x, iter, tolerance=0.00001):
    classifiers = classifier_set()

    for i in range(1, iter):
        print("Iteration", i)
        svm = svm_train(record_weights, y, x, '-t 0')
        c = classifier(svm)
        p_labels = np.array(c.get_predicted_label(y, x)).astype(np.float64)

        err = 0
        for i in range(len(y)):
            if p_labels[i] != y[i]:
                err += record_weights[i]

        c.update_weight(err)
        classifiers.add(c)

        if err < tolerance:
            break

        record_weights = record_weights * np.exp(-c.weight * np.array(y) * p_labels)
        record_weights /= np.sum(record_weights)


    return classifiers

def train(K):
    y, x = svm_read_problem('DogsVsCats/DogsVsCats.train')

    w = np.ones(len(y))/(len(y))
    
    print('Training AdaBoost K =', K, '...')
    ensemble = adaboost(w, y, x, K)

    print('Testing...')
    y_test, x_test = svm_read_problem('DogsVsCats.test')
    p_pred, accuracy = ensemble.predict(y_test, x_test)
    print('K = ', K, 'Accuracy: ', accuracy)


if __name__ == "__main__":
    train(10)
    train(20)

        
