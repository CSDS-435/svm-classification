# svm-classification
An image classifier using SVM and the Adaboost algorithm. 

Data is pulled from a competition to create an algorithm that distinguishes dogs from cats on Kaggle ([link](https://www.kaggle.com/c/dogs-vs-cats)), and preprocessed into 64-color histograms of the corresponding image. The values of the features are normalized fraction of pixels in the image of a given color bin.

To cross-validate and implement Adaboost, LIBSVM was used ([link] (https://www.csie.ntu.edu.tw/~cjlin/libsvm/))

Adaboost is tested with 10 and 20 iterations, and gives an accuracy of around 60%. This is somewhat expected, as it is a weak classifier, and shows that there is room for improvement of the algorithm.
