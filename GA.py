import numpy
import random
import NeuralNet as NN
from tqdm import tqdm
import time

def geneticAlgorithm(populationSize, crossoverRate, mutationRate, neuronsPerLayer, maxGenerations, dataInputs, dataOutputs, printStatus, metaGA, timeStoppingCriterion, maxTime):  
    if (printStatus):
        print("Setting up GA...")
    # Initial population generation
    population = []
    for network in range(populationSize):
        #define neural network object
        model = NN.NeuralNet()
        for layer in numpy.arange(0, len(neuronsPerLayer)):
            # Input layer, Weights for first hidden layer 
            if (layer == 0):
                model.addLayer(numpy.random.uniform(low=-0.1, high=0.1, size=(dataInputs.shape[1],neuronsPerLayer[0])))
                #Add bias vector for this layer, range of values can be played around with
                model.addBiasVector(numpy.random.uniform(low=-0.1, high=0.1, size=(neuronsPerLayer[layer])))
            else:
                model.addLayer(numpy.random.uniform(low=-0.1, high=0.1, size=(neuronsPerLayer[layer-1],neuronsPerLayer[layer])))
                #Add bias vector for this layer, range of values can be played around with
                model.addBiasVector(numpy.random.uniform(low=-0.1, high=0.1, size=(neuronsPerLayer[layer])))
    
        population.append(model)
        
    # Main generational loop
    if (printStatus):
        print("Training NN with GA:")
        print("")
        pbar = tqdm(total=maxGenerations)
    accuracies = []
    startTime = time.time()
    for generation in range(maxGenerations):
        if (timeStoppingCriterion):
            if ((time.time() - startTime) > maxTime):
                break
        
        fitness = []
        for network in range(populationSize):
            fitness.append(population[network].evaluate(dataInputs, dataOutputs))
            
        accuracies.append(fitness[0])
        
        parents = selection(population.copy(), fitness.copy(), crossoverRate)
        offspring = generateOffspring(parents, populationSize, mutationRate)
        
        
        population = []
        population.extend(parents)
        population.extend(offspring)
        
        if (printStatus):
            pbar.update(1)
        
    if (metaGA):
        return accuracies[len(accuracies) - 1]
    else:
        return accuracies, population[0]

def selection(networks, fitness, crossoverRate):
    parents = []
    for parent in range(int(crossoverRate*len(networks))):
        parentId = numpy.where(fitness == numpy.max(fitness))
        parents.append(networks[parentId[0][0]])
        del networks[parentId[0][0]]
        del fitness[parentId[0][0]]
    return parents

def crossover(network1, network2):
    vectorWeights1 = matToVector(numpy.array([network1.layers]))[0]
    vectorWeights2 = matToVector(numpy.array([network2.layers]))[0]
    
    #Get Bias vectors in form of one array
    biasVector1= numpy.concatenate(network1.biasVectors, axis=0)
    biasVector2= numpy.concatenate(network2.biasVectors, axis=0)
    
    #Crossover two bias vectors
    biasVectorOffspring=[]
    biasVectorCrossoverPoint=numpy.uint32(len(biasVector1/2))
    biasVectorOffspring.extend(biasVector1[:biasVectorCrossoverPoint])
    biasVectorOffspring.extend(biasVector2[biasVectorCrossoverPoint:])
    
    
    # SingleMidPointCrossover
    crossoverPoint = numpy.uint32(len(vectorWeights1)/2)
    offspring = []
    offspring.extend(vectorWeights1[:crossoverPoint])
    offspring.extend(vectorWeights2[crossoverPoint:])
    return offspring, biasVectorOffspring
    
def mutate(weightsInVectorForm,biasVector, mutationRate):
    mutationIndicies = numpy.array(random.sample(range(0, len(weightsInVectorForm)), int(mutationRate*len(weightsInVectorForm))))
    biasVecMutationIndicies = numpy.array(random.sample(range(0, len(biasVector)), int(mutationRate*len(biasVector))))

    for index in mutationIndicies:
        weightsInVectorForm[index] = weightsInVectorForm[index] + numpy.random.uniform(-1.0, 1.0, 1)[0]
        
    for biasIndex in biasVecMutationIndicies:
        biasVector[biasIndex]= biasVector[biasIndex]+numpy.random.uniform(-1.0, 1.0, 1)[0]
    return weightsInVectorForm, biasVector

def generateOffspring(parents, populationSize, mutationRate):
    offspring = []
    for i in range(populationSize - len(parents)):
        # Randomly select 2 parents
        parentIndicies = random.sample(range(0,len(parents)), 2)
        weightsOffSpring, biasVecOffSpring=crossover(parents[parentIndicies[0]], parents[parentIndicies[1]])
        newWeights, newBiasVec = mutate(weightsOffSpring, biasVecOffSpring, mutationRate)
        offspringMatrix = vectorToMat(numpy.array([newWeights]), numpy.array([parents[0].layers]))[0]
        offspringNN = NN.NeuralNet()
        offSpringBiasVectors=splitBiasVector(newBiasVec, offspringMatrix)
        offspringNN.setLayers(offspringMatrix)
        offspringNN.setBiasVectors(offSpringBiasVectors)
        offspring.append(offspringNN)
    return offspring
        
def weightsMatrixToKerasWeights(matrixWeights):
    kerasWeights = []
    for i in range(len(matrixWeights)):
        kerasWeights.append(matrixWeights[i])
        kerasWeights.append(numpy.zeros(len(matrixWeights[i][0])))
    return kerasWeights

def kerasWeightsToWeightsMatrix(network):
    networkWeights = network.get_weights()
    weightsMatrix = []
    for i in range(int(len(networkWeights)/2)):
        weightsMatrix.append(networkWeights[i*2])
    return weightsMatrix

def splitBiasVector(biasVector, weightsMatrix):
    biasVectors=[]
    for layer in weightsMatrix:
        nodesInLayer= len(layer[1])
        biasVecForLayer=biasVector[:nodesInLayer]
        biasVectors.append(biasVecForLayer)
        del biasVector[:nodesInLayer]
    return biasVectors

# ==============================================================
# Next 2 functions taken directly from internet
# ==============================================================

# Converting each solution from matrix to vector.
def matToVector(mat_pop_weights):
    pop_weights_vector = []
    for sol_idx in range(mat_pop_weights.shape[0]):
        curr_vector = []
        for layer_idx in range(mat_pop_weights.shape[1]):
            # I think this just gets all the weights from a layer and puts them into a 1d array
            vector_weights = numpy.reshape(mat_pop_weights[sol_idx, layer_idx], newshape=(mat_pop_weights[sol_idx, layer_idx].size))
            curr_vector.extend(vector_weights)
        pop_weights_vector.append(curr_vector)
    return numpy.array(pop_weights_vector)

# Converting each solution from vector to matrix.
def vectorToMat(vector_pop_weights, mat_pop_weights):
    mat_weights = []
    for sol_idx in range(mat_pop_weights.shape[0]):
        start = 0
        end = 0
        for layer_idx in range(mat_pop_weights.shape[1]):
            end = end + mat_pop_weights[sol_idx, layer_idx].size
            curr_vector = vector_pop_weights[sol_idx, start:end]
            mat_layer_weights = numpy.reshape(curr_vector, newshape=(mat_pop_weights[sol_idx, layer_idx].shape))
            mat_weights.append(mat_layer_weights)
            start = end
    return numpy.reshape(mat_weights, newshape=mat_pop_weights.shape)