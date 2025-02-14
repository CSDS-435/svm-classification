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

# train by adaboost
# takes weights of record, problem input, number of iterations, and tolarance level
# returns set of updated classifiers
def adaboost(y, x, iter, tolerance=0.00001):
    classifiers = []
    weights = np.ones(len(y))/(len(y))

    for i in range(1, iter+1):
        print("Iteration", i)
        prob = svm_problem(weights, y, x)
        param = svm_parameter('-t 0 -h 0 -q')
        svm = svm_train(prob, param)

        c = classifier(svm)

        p_labels, p_acc, p_vals = svm_predict(y, x, c.svm)
        p_labels = np.array(p_labels).astype(np.float64)

        err = 0
        for i in range(len(y)):
            if p_labels[i] != y[i]:
                err += weights[i]
        
        c.update_weight(err)
        classifiers.append(c)

        if err < tolerance:
            break

        weights = weights * np.exp(-c.weight * np.array(y) * p_labels)
        weights /= np.sum(weights)

    return classifiers

# predict
# takes set of classifiers and problem input
# return array of prediction and percentage of accuracy
# 1 - dog, -1 - cat
def predict(classifiers, y, x):
    predictions = [0] * len(x)
    for c in classifiers:
        p_labels, p_acc, p_vals = svm_predict([], x, c.svm, '-q')
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
    
    accuracy = matches/len(x) * 100
    return predictions, accuracy

if __name__ == "__main__":
    y, x = svm_read_problem('DogsVsCats/DogsVsCats.train')

    print("Training AdaBoost K = 10...")
    classifiers = adaboost(y, x, 10)

    print('Testing...')
    y_test, x_test = svm_read_problem('DogsVsCats/DogsVsCats.test')
    prediction, accuracy = predict(classifiers, y_test, x_test)
    print("K = 10", "Accuracy:", accuracy)
    
    y, x = svm_read_problem('DogsVsCats/DogsVsCats.train')

    print("Training AdaBoost K = 20...")
    classifiers = adaboost(y, x, 20)

    print('Testing...')
    y_test, x_test = svm_read_problem('DogsVsCats/DogsVsCats.test')
    prediction, accuracy = predict(classifiers, y_test, x_test)
    print("K = 20", "Accuracy:", accuracy)
