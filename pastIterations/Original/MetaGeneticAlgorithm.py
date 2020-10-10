# Saved results
#{'populationSize': 10.983210513716132, 'crossoverRate': 0.39608837688777504, 'mutationRate': 0.0783912860245263}
#{'populationSize': 11, 'crossoverRate': 0.39608837688777504, 'mutationRate': 0.0783912860245263}
#{'populationSize': 11, 'crossoverRate': 0.39608837688777504, 'mutationRate': 0.0783912860245263}
#{'populationSize': 10.940523401449232, 'crossoverRate': 0.4753804990414217, 'mutationRate': 0.0783912860245263}
#{'populationSize': 11, 'crossoverRate': 0.3298728453621818, 'mutationRate': 0.10617611936456836}
#{'populationSize': 10.992070882830586, 'crossoverRate': 0.44967661979358087, 'mutationRate': 0.0783912860245263}

import random
import time
from sklearn import datasets
import numpy
import GeneticAlgorithm as ga
import NeuralNetwork as ANN
from tqdm import tqdm

sklearnDataset = datasets.load_breast_cancer()
data_inputs = numpy.array(sklearnDataset.data)
data_outputs = numpy.array(sklearnDataset.target)
noOfClasses = len(numpy.unique(data_outputs))

def initialisePopulation(populationSize):
    population = []
    for i in range(populationSize):
        parameters = {}
        parameters["populationSize"] = int(random.uniform(5,15))
        parameters["crossoverRate"] = random.uniform(0.25,0.75)
        parameters["mutationRate"] = random.uniform(0.05,0.15)
        population.append(parameters)
    return population
    
def calcTerminalTimeCondition(currentTime, startTime, runTime):
    print(str(runTime - (currentTime - startTime)) + " seconds left")
    return ((currentTime - startTime) > runTime)

def geneticAlgorithm(networkParameters, runs):
    accuracies = []
    for run in range(runs):
        maxGenerations = 30
        neuronsPerLayer = [150,60,noOfClasses]
        initialPopulationWithWeightsInMatrixForm = []    
        for network in numpy.arange(0, networkParameters["populationSize"]):
            currentNetworkWeights = []
            for layer in numpy.arange(0, len(neuronsPerLayer)):
                if (layer == 0):
                    currentNetworkWeights.append(numpy.random.uniform(low=-0.1, high=0.1,
                                                                      size=(data_inputs.shape[1],neuronsPerLayer[0])))
                else:
                    currentNetworkWeights.append(numpy.random.uniform(low=-0.1, high=0.1,
                                                                      size=(neuronsPerLayer[layer-1],neuronsPerLayer[layer])))
            initialPopulationWithWeightsInMatrixForm.append(numpy.array(currentNetworkWeights))
        
        # Just converts to numpy array
        populationWithWeightsInMatrixForm = numpy.array(initialPopulationWithWeightsInMatrixForm)
        populationWithWeightsInVectorForm = ga.mat_to_vector(populationWithWeightsInMatrixForm)
        
        for generation in range(maxGenerations):
            # converting the solutions from being vectors to matrices.
            populationWithWeightsInMatrixForm = ga.vector_to_mat(populationWithWeightsInVectorForm, 
                                               populationWithWeightsInMatrixForm)
        
            # Measuring the fitness of each chromosome in the population.
            fitness = ANN.fitness(populationWithWeightsInMatrixForm, 
                                  data_inputs, 
                                  data_outputs, 
                                  activation="sigmoid")
        
            # Selecting the best parents in the population for mating.
            parents = ga.select_mating_pool(populationWithWeightsInVectorForm, 
                                            fitness.copy(), 
                                            int(networkParameters["crossoverRate"]*networkParameters["populationSize"]))
        
            # Generating next generation using crossover.
            offspring_crossover = ga.crossover(parents,
                                               # offspring(populationSize-numParents,totalConnections)
                                               offspring_size=(populationWithWeightsInVectorForm.shape[0]-parents.shape[0], populationWithWeightsInVectorForm.shape[1]))
            
            # Adding some variations to the offsrping using mutation.
            offspring_mutation = ga.mutation(offspring_crossover, 
                                             mutation_percent=int(networkParameters["mutationRate"]*100))
        
            # Creating the new population based on the parents and offspring.
            populationWithWeightsInVectorForm[0:parents.shape[0], :] = parents
            populationWithWeightsInVectorForm[parents.shape[0]:, :] = offspring_mutation
        
        populationWithWeightsInMatrixForm = ga.vector_to_mat(populationWithWeightsInVectorForm, populationWithWeightsInMatrixForm)
        best_weights = populationWithWeightsInMatrixForm [0, :]
        acc, predictions = ANN.predict_outputs(best_weights, data_inputs, data_outputs, activation="sigmoid")
        accuracies.append(acc)
    return numpy.median(accuracies)

def crossover(parameterSet1, parameterSet2):
    # single point midway crossover
    crossoverPoint = numpy.uint32(len(parameterSet1)/2)
    newParameterSet = {}
    counter = 0
    for key,  value in parameterSet1.items():
        if (counter < crossoverPoint):
            newParameterSet[key] = value
        else:
            newParameterSet[key] = parameterSet2[key]
        counter = counter + 1
    return newParameterSet

def mutate(parameterSet, mutationChance):
    for key, value in parameterSet.items():
        if (random.uniform(0,1) <= mutationChance):
            if (key == "populationSize"):
                parameterSet[key] = parameterSet[key] + random.uniform(-5,5)
            elif (key == "crossoverRate"):
                parameterSet[key] = parameterSet[key] + random.uniform(-0.1,0.1)
            else:
                parameterSet[key] = parameterSet[key] + random.uniform(-0.02,0.02)
                
    return parameterSet

def evolvePopulation(population, fitness, crossoverRate, mutationChance):
    parentsForBreeding = []
    populationSize = len(population)
    for parent in range(int(crossoverRate*len(population))):
        parentId = numpy.where(fitness == numpy.max(fitness))
        parentsForBreeding.append(population[parentId[0][0]])
        population.remove(population[parentId[0][0]])
        fitness.remove(fitness[parentId[0][0]])
    newGeneration = parentsForBreeding.copy()
    for offspring in range(populationSize - len(newGeneration)):
        parentIds = random.sample(range(0,len(parentsForBreeding)),2)
        newGeneration.append(mutate(crossover(parentsForBreeding[parentIds[0]], parentsForBreeding[parentIds[1]]), mutationChance))
    return newGeneration
        
populationSize = 8
population = initialisePopulation(populationSize)
crossoverRate = 0.5
mutationChance = 0.33
runTime = 1200
runs = 3
terminalCondition = False
startTime = time.time()
generation = 0
while (not terminalCondition):
    print("Generation " + str(generation))
    fitness = []
    #pbar = tqdm(total=len(population))
    for paramset in population:
        #pbar.update(1)
        fitness.append(geneticAlgorithm(paramset, runs))
    #print("")
    population = evolvePopulation(population, fitness, crossoverRate, mutationChance)
    terminalCondition = calcTerminalTimeCondition(time.time(), startTime, runTime)
    generation = generation + 1

print("Generation " + str(generation) + " best parameter set: ")

# Find best out of final population
bestFitness = 0
for paramset in population:
    paramsetFitness = geneticAlgorithm(paramset, runs)
    if (paramsetFitness > bestFitness):
        bestFitness = paramsetFitness
        bestParamset = paramset

print(bestParamset)