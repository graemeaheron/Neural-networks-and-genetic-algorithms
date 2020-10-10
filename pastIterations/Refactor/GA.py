import numpy
import random

def selection(networks, fitness, crossoverRate):
    parents = []
    for parent in range(int(crossoverRate*len(networks))):
        parentId = numpy.where(fitness == numpy.max(fitness))
        parents.append(networks[parentId[0][0]])
        del networks[parentId[0][0]]
        del fitness[parentId[0][0]]
    return parents, networks

def crossover(network1, network2):
    vectorWeights1 = matToVector(numpy.array([kerasWeightsToWeightsMatrix(network1)]))[0]
    vectorWeights2 = matToVector(numpy.array([kerasWeightsToWeightsMatrix(network2)]))[0]
    # SingleMidPointCrossover
    crossoverPoint = numpy.uint32(len(vectorWeights1)/2)
    offspring = []
    offspring.extend(vectorWeights1[:crossoverPoint])
    offspring.extend(vectorWeights2[crossoverPoint:])
    return offspring
    
def mutate(weightsInVectorForm, mutationRate):
    mutationIndicies = numpy.array(random.sample(range(0, len(weightsInVectorForm)), int(mutationRate*len(weightsInVectorForm))))
    for index in mutationIndicies:
        weightsInVectorForm[index] = weightsInVectorForm[index] + numpy.random.uniform(-1.0, 1.0, 1)[0]
    return weightsInVectorForm

def generateOffspring(parents, waste, mutationRate):
    # This could do with fixing but it's so I can use the vecToMat and MatToVec functions off internet
    tempSaveForSize = numpy.array([kerasWeightsToWeightsMatrix(parents[0])])
    for i in range(len(waste)):
        # Randomly select 2 parents
        parentIndicies = random.sample(range(0,len(parents)), 2)
        newWeights = mutate(crossover(parents[parentIndicies[0]], parents[parentIndicies[1]]), mutationRate)
        # This could do with fixing but it's so I can use the vecToMat and MatToVec functions off internet
        offspringMatrix = vectorToMat(numpy.array([newWeights]),tempSaveForSize)[0]
        waste[i].set_weights(weightsMatrixToKerasWeights(offspringMatrix))
    return waste
        
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