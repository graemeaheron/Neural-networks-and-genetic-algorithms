import numpy
import random

class Sample:
    def __init__(self, features, target):
        self.features = features
        self.target = target
    
    def getFeatures(self):
        return self.features
    
    def getTarget(self):
        return self.target
    
class SampleSet:
    def __init__(self, data, target):
        self.data = data
        self.target = target

class testAndTrainingSets:
    def __init__(self, trainingData, trainingTarget, testingData, testingTarget):
        self.training = SampleSet(trainingData, trainingTarget)
        self.testing = SampleSet(testingData, testingTarget)
        
def splitSamples(samples, noOfTestPoints):
    samplesData = samples.data.copy()
    samplesTarget = samples.target.copy()
    
    samples = []
    for i in range(0, len(samplesData)):
        samples.append(Sample(
            samplesData[i],
            samplesTarget[i]))
    
    random.shuffle(samples)

    testIndicies = random.sample(range(0, len(samples)), noOfTestPoints)

    trainingData = []
    trainingTarget = []
    testingData = []
    testingTarget = []

    for i in numpy.arange(0, len(samplesData)):
        if (i in testIndicies):
            testingData.append(samples[i].getFeatures())
            testingTarget.append(samples[i].getTarget())
        else:
            trainingData.append(samples[i].getFeatures())
            trainingTarget.append(samples[i].getTarget())
            
    return testAndTrainingSets(trainingData,trainingTarget,testingData,testingTarget)

def calculateErrors(predicitions, actual):
    errors = 0
    for i in range(0, len(predicitions)):
        if (predicitions[i] != actual[i]):
            errors = errors + 1
    return errors