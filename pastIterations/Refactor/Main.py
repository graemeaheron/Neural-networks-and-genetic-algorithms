import numpy
import matplotlib.pyplot
from sklearn import datasets
from tqdm import tqdm
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import NeuralNet as NN
import GA

# May need to change max generations
def geneticAlgorithm(populationSize, crossoverRate, mutationRate, neuronsPerLayer, maxGenerations, dataInputs, dataOutputs):
    print("Converting data for GA evaluation using Keras...")
    dataOutputsKeras = to_categorical(dataOutputs, len(numpy.unique(dataOutputs)))
    
    print("Setting up GA (Keras compilation)...")
    # Initial population generation
    population = []
    for network in range(populationSize):
        model = Sequential()
        for layer in numpy.arange(0, len(neuronsPerLayer)):
            # Input layer
            if (layer == 0):
                model.add(Dense(units=neuronsPerLayer[layer], activation="sigmoid", input_shape=(dataInputs.shape[1],)))
                model.layers[layer].set_weights(
                        [numpy.random.uniform(low=-0.1, high=0.1, size=(dataInputs.shape[1],neuronsPerLayer[0])),
                         numpy.zeros(neuronsPerLayer[layer])
                         ])
            else:
                model.add(Dense(units=neuronsPerLayer[layer], activation="sigmoid"))
                model.layers[layer].set_weights(
                        [numpy.random.uniform(low=-0.1, high=0.1, size=(neuronsPerLayer[layer-1],neuronsPerLayer[layer])),
                         numpy.zeros(neuronsPerLayer[layer])
                         ])
    
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        population.append(model)
        
    for network in range(populationSize):
        population[network].evaluate(verbose=0, x=dataInputs, y=dataOutputsKeras)
        
    # Main generational loop
    print("Training NN with GA:")
    print("")
    accuracies = []
    pbar = tqdm(total=maxGenerations)
    for generation in range(maxGenerations):
        
        fitness = []
        for network in range(populationSize):
            fitness.append(population[network].evaluate(verbose=0, x=dataInputs, y=dataOutputsKeras)[1] * 100)
            
        accuracies.append(fitness[0])
        
        parents, waste = GA.selection(population.copy(), fitness.copy(), crossoverRate)
        offspring = GA.generateOffspring(parents, waste, mutationRate)
        
        
        population = []
        population.extend(parents)
        population.extend(offspring)
        
        pbar.update(1)
        
    return accuracies, population[0]
        
# Get datasets
sklearnDataset = datasets.load_breast_cancer()
dataInputs = numpy.array(sklearnDataset.data)
dataOutputs = numpy.array(sklearnDataset.target)
noOfClasses = len(numpy.unique(dataOutputs))
        
acc, network = geneticAlgorithm(8, 0.5, 0.1, [20,20,noOfClasses], 100, dataInputs, dataOutputs)

print("Best accuracy: " + str(acc[len(acc)]))