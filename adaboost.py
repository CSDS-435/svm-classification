import sys
sys.path.append("/Users/minhqpham/Documents/CWRU/Spring 2021/CSDS 435/libsvm-weights/python")
# print(sys.path)

from svmutil import *
from commonutil import *
import numpy as np

# class for a single classifier, holds information about its svm and weight
class classifier:
    def __init__(self, svm, weight=0):
        self.svm = svm
        self.weight = weight

    def update_weight(self, err):
        if err == 0:
            self.weight = 10000.0
        self.weight = 0.5 * np.log((1.0-err)/err)

# class for set of classifiers
class classifier_set:
    def __init__(self):
        self.classifiers = []
    # predict
    # takes problem input, return array of prediction and percentage of accuracy
    # 1 - dog, -1 - cat
    def predict(self, y, x):
        predictions = [0] * len(x)
        for c in self.classifiers:
            p_labels, p_acc, p_vals = svm_predict(y, x, c.svm)
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

# train by adaboost
# takes weights of record, problem input, number of iterations, and tolarance level
# returns set of updated classifiers
def adaboost(record_weights, y, x, iter, tolerance=0.00001):
    classifiers = classifier_set()

    for i in range(1, iter):
        print("Iteration", i)
        svm = svm_train(record_weights, y, x, '-t 0 -h 0')
        c = classifier(svm)
        p_labels, p_acc, p_vals = svm_predict(y, x, c.svm)
        p_labels = np.array(p_labels).astype(np.float64)

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

if __name__ == "__main__":
    y, x = svm_read_problem('DogsVsCats/DogsVsCats.train')

    w = np.ones(len(y))/(len(y))
    
    print("Training AdaBoost K = 10...")
    ensemble = adaboost(w, y, x, 10)

    print('Testing...')
    y_test, x_test = svm_read_problem('DogsVsCats/DogsVsCats.test')
    prediction, accuracy = ensemble.predict(y_test, x_test)
    print("K = 10", "Accuracy:", accuracy)

    print("Training AdaBoost K = 20...")
    ensemble = adaboost(w, y, x, 20)

    print('Testing...')
    y_test, x_test = svm_read_problem('DogsVsCats/DogsVsCats.test')
    prediction, accuracy = ensemble.predict(y_test, x_test)
    print("K = 20", "Accuracy:", accuracy)

        
