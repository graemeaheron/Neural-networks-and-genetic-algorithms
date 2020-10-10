from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import time
from datasetLoading import loadBreastCancer, loadDigits
import helper
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

dataInputs, dataOutputs = loadBreastCancer()
dataInputs = numpy.array(dataInputs)
dataOutputs = numpy.array(dataOutputs)
noOfClasses = len(numpy.unique(dataOutputs))

kf = KFold(n_splits=5)
kf.get_n_splits(dataInputs)

kFoldAccuracies=[]
accuracies=[]
kFoldTimes = []
iteration=0

Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
gammas = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = {'C': Cs, 'gamma' : gammas}


grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=10)

grid.fit(dataInputs, dataOutputs)

print("The best classifier is: ", grid.best_estimator_)


for train_index , test_index in kf.split(dataInputs):
    iteration+=1
    print("iteration", iteration)
    X_train, X_test= dataInputs[train_index], dataInputs[test_index]
    y_train, y_test= dataOutputs[train_index], dataOutputs[test_index]
    #Transform data 
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    startTime = time.time()
    supportVectorMachine = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    supportVectorMachine.fit(X_train, y_train)
    endTime = time.time()
    predictions = supportVectorMachine.predict(X_test)
    acc = 100 - 100*helper.calculateErrors(predictions, y_test)/len(predictions)
    accuracies.append(acc)
    print("Accuracy on testing data for this fold: ", acc)
    kFoldAccuracies.append(acc)
    kFoldTimes.append(float(endTime-startTime))


meanAccuracy = sum(kFoldAccuracies)/len(kFoldAccuracies)  

print ("Mean testing Accuracy: ", meanAccuracy)