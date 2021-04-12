import sys
sys.path.append("/Users/minhqpham/Documents/CWRU/Spring 2021/CSDS 435/libsvm/python")
print(sys.path)

from svmutil import *
from commonutil import *

y, x = svm_read_problem('DogsVsCats/DogsVsCats.train')
prob = svm_problem(y, x)

for x in range(1,10):
    # linear
    print("10-fold Linear Kernel - iteration #%d" % (x))
    param = svm_parameter('-t 0 -v 10 -h 0')
    m = svm_train(prob, param)
    # polynomial with degree 5
    print("\n10-fold Polynomial Kernel - iteration #%d" % (x))
    param = svm_parameter('-t 1 -d 5 -v 10 -h 0')
    m = svm_train(prob, param)

# training dataset
print("\nTraining dataset - Linear Kernel")
param = svm_parameter('-t 0 -h 0')
m = svm_train(prob, param)
p_labels, p_acc, p_vals = svm_predict(y, x, m)

print("\nTraining dataset - Polynomial Kernel")
param = svm_parameter('-t 1 -d 5 -h 0')
m = svm_train(prob, param)
p_labels, p_acc, p_vals = svm_predict(y, x, m)

# testing dataset
y, x = svm_read_problem('DogsVsCats/DogsVsCats.test')

print("\nTesting dataset - Linear Kernel")
param = svm_parameter('-t 0 -h 0')
m = svm_train(prob, param)
p_labels, p_acc, p_vals = svm_predict(y, x, m)

print("\nTesting dataset - Polynomial Kernel")
param = svm_parameter('-t 1 -d 5 -h 0')
m = svm_train(prob, param)
p_labels, p_acc, p_vals = svm_predict(y, x, m)
