import numpy
import matplotlib.pyplot
from GA import geneticAlgorithm
from datasetLoading import loadBreastCancer, loadDigits
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import time
        
dataInputs, dataOutputs = loadBreastCancer()
dataInputs = numpy.array(dataInputs)
dataOutputs = numpy.array(dataOutputs)
noOfClasses = len(numpy.unique(dataOutputs))

std = StandardScaler()

kf = KFold(n_splits=5)
kf.get_n_splits(dataInputs)

kFoldAccuracies=[]
kFoldTimes = []
iteration=0

for train_index , test_index in kf.split(dataInputs):
    iteration+=1
    print("iteration", iteration)
    X_train, X_test= dataInputs[train_index], dataInputs[test_index]
    y_train, y_test= dataOutputs[train_index], dataOutputs[test_index]
    #Transform data 
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    startTime = time.time()
    acc, network = geneticAlgorithm(8, 0.5, 0.1, [20,20,noOfClasses], 200, X_train, y_train, False, False, False, 0)
    endTime = time.time()
    print("Accuracy on Training data", str(acc[len(acc) - 1]))
    accuracyTest=network.evaluate(X_test, y_test)
    print("Accuracy on testing data for this fold: ", accuracyTest)
    kFoldAccuracies.append(accuracyTest)
    kFoldTimes.append(float(endTime-startTime))

meanAccuracy = sum(kFoldAccuracies)/len(kFoldAccuracies)  

print ("Mean testing Accuracy: ", meanAccuracy)