import numpy
import GeneticAlgorithm as ga
import NeuralNetwork as ANN
import matplotlib.pyplot
from sklearn import datasets
import helper
from tqdm import tqdm
import time

sklearnDataset = datasets.load_breast_cancer()
data_inputs = numpy.array(sklearnDataset.data)
data_outputs = numpy.array(sklearnDataset.target)
noOfClasses = len(numpy.unique(data_outputs))

"""
Genetic algorithm parameters:
    Mating Pool Size (Number of Parents)
    Population Size
    Number of Generations
    Mutation Percent
"""

# Solutions per population
# Original:
#networksPerPopulation = 8
#num_parents_mating = 4
#maxGenerations = 100
#featureMutationPrecentage = 10
#neuronsPerLayer = [150,60,noOfClasses]
networksPerPopulation = 8
num_parents_mating = 4
maxGenerations = 100
featureMutationPrecentage = 10
neuronsPerLayer = [20,20,noOfClasses]

#Creating the initial population.
initialPopulationWithWeightsInMatrixForm = []    
for network in numpy.arange(0, networksPerPopulation):
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

best_outputs = []
accuracies = numpy.empty(shape=(maxGenerations))

print("Training NN with genetic algorithm:")
pbar = tqdm(total=maxGenerations)
nnPreLearningTime = time.time()

for generation in range(maxGenerations):
    # converting the solutions from being vectors to matrices.
    populationWithWeightsInMatrixForm = ga.vector_to_mat(populationWithWeightsInVectorForm, 
                                       populationWithWeightsInMatrixForm)

    # Measuring the fitness of each chromosome in the population.
    fitness = ANN.fitness(populationWithWeightsInMatrixForm, 
                          data_inputs, 
                          data_outputs, 
                          activation="sigmoid")
    accuracies[generation] = fitness[0]

    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(populationWithWeightsInVectorForm, 
                                    fitness.copy(), 
                                    num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents,
                                       # offspring(populationSize-numParents,totalConnections)
                                       offspring_size=(populationWithWeightsInVectorForm.shape[0]-parents.shape[0], populationWithWeightsInVectorForm.shape[1]))
    
    # Adding some variations to the offsrping using mutation.
    offspring_mutation = ga.mutation(offspring_crossover, 
                                     mutation_percent=featureMutationPrecentage)

    # Creating the new population based on the parents and offspring.
    populationWithWeightsInVectorForm[0:parents.shape[0], :] = parents
    populationWithWeightsInVectorForm[parents.shape[0]:, :] = offspring_mutation
    
    pbar.update(1)

nnPostLearningTime = time.time()

populationWithWeightsInMatrixForm = ga.vector_to_mat(populationWithWeightsInVectorForm, populationWithWeightsInMatrixForm)
best_weights = populationWithWeightsInMatrixForm [0, :]
acc, predictions = ANN.predict_outputs(best_weights, data_inputs, data_outputs, activation="sigmoid")

matplotlib.pyplot.plot(accuracies, linewidth=2, color="black")
matplotlib.pyplot.xlabel("Iteration", fontsize=10)
matplotlib.pyplot.ylabel("Fitness", fontsize=10)
matplotlib.pyplot.xticks(numpy.arange(0, maxGenerations+1, 100), fontsize=10)
matplotlib.pyplot.yticks(numpy.arange(0, 101, 5), fontsize=10)

print("")
print("Accuracy of the best solution using GA: ", acc)
print("NN learning time: " + str(nnPostLearningTime -  nnPreLearningTime))

# =============================================================================
# Comparing to Keras to check correctness of accuracy
# =============================================================================

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

data_outputs_keras = to_categorical(data_outputs, noOfClasses)

model = Sequential()

for layer in numpy.arange(0, len(neuronsPerLayer)):
    if (layer == 0):
        model.add(Dense(units=neuronsPerLayer[layer], activation="sigmoid", input_shape=(data_inputs.shape[1],)))
    else:
        model.add(Dense(units=neuronsPerLayer[layer], activation="sigmoid"))
    model.layers[layer].set_weights(numpy.array([best_weights[layer], numpy.zeros(neuronsPerLayer[layer])]))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

print("Evaluating with Keras:")
print("Accuracy of same solution in keras (should be the same):" + str(model.evaluate(x=data_inputs, y=data_outputs_keras)[1] * 100))

# =============================================================================
# RBF SVM comparison (will need redoing to so same training and testing data
# used for SVM and NN)
# =============================================================================

from sklearn import svm
splitResult = helper.splitSamples(sklearnDataset,10)

supportVectorMachine = svm.SVC()
print("Training SVM....")
svmPreLearningTime = time.time()
supportVectorMachine.fit(X=splitResult.training.data,y=splitResult.training.target)
svmPostLearningTime = time.time()
svmPredicitions = supportVectorMachine.predict(splitResult.testing.data)
print("Evaluating with SVM...")
print("Accuracy of RBF SVM: " + str(100 - 100*(helper.calculateErrors(svmPredicitions, splitResult.testing.target)/len(svmPredicitions))))
print("SVM learning time: " + str(svmPostLearningTime-svmPreLearningTime))