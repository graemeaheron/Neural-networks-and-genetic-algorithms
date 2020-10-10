import numpy
import warnings
warnings.filterwarnings("ignore")

class NeuralNet:
    
    def __init__(self):
        self.layers = []
        self.biasVectors=[]
        
    def addLayer(self, weights):
        self.layers.append(weights)
        
    def addBiasVector(self, biasVector):
        self.biasVectors.append(biasVector)
    
    def setLayers(self, layers):
        self.layers = layers
        
    def setBiasVectors(self, biasVectors):
        self.biasVectors = biasVectors
    
    def sigmoid(self, value):
        return 1.0/(1.0+numpy.exp(-1*value))
    
    def evaluate(self, dataInputs, dataOutputs):
        predictions = []
        for sample in range(len(dataInputs)):
            result = dataInputs[sample, :]
            for i in range(len(self.layers)):
                result = numpy.matmul(result, self.layers[i])
                #Add Bias Vector for this layer
                result = result + self.biasVectors[i]  
                result = self.sigmoid(result)
            predictedLabel = numpy.where(result == numpy.max(result))[0][0]
            predictions.append(predictedLabel)
        return 100 * numpy.where(predictions == dataOutputs)[0].size / dataOutputs.size