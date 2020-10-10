import csv

def loadBreastCancer():
    data = open("datasets/wdbc.data")
    lines = data.readlines()
    dataInputs = []
    dataOutputs = []
    for line in lines:
        splitData = line.split(",")
        currentSample = []
        for i in range(len(splitData)):
            if (i == 1):
                dataOutputs.append(splitData[i])
            elif (i > 1):
                currentSample.append(float(splitData[i]))
        dataInputs.append(currentSample)
        # 0 is M, 1 is B
        for i in range(len(dataOutputs)):
            if (dataOutputs[i] == "M"):
                dataOutputs[i] = 0
            elif (dataOutputs[i] == "B"):
                dataOutputs[i] = 1
    data.close()
    return dataInputs, dataOutputs

def loadDigits():
    data = open("datasets/optdigits.data")
    lines = data.readlines()
    dataInputs = []
    dataOutputs = []
    for line in lines:
        splitData = line.split(",")
        currentSample = []
        for i in range(len(splitData)):
            if (i == 64):
                dataOutputs.append(int(splitData[i]))
            else :
                currentSample.append(int(splitData[i]))
        dataInputs.append(currentSample)
    data.close()
    return dataInputs, dataOutputs

def loadParkinsons():
    data = open("datasets/park_train_data.txt")
    lines = data.readlines()
    dataInputs = []
    dataOutputs = []
    counter = 0
    for line in lines:
        splitData = line.split(",")
        currentSample = []
        for i in range(len(splitData)):
            if ((i == 27) | (i == 0)):
                #Nothing
                counter = counter + 1
            elif (i == 28):
                dataOutputs.append(float(splitData[i]))
            else:
                currentSample.append(float(splitData[i]))
        dataInputs.append(currentSample)
    data.close()
    data = open("datasets/park_test_data.txt")
    lines = data.readlines()
    counter = 0
    for line in lines:
        splitData = line.split(",")
        currentSample = []
        for i in range(len(splitData)):
            if ((i == 0)):
                #Nothing
                counter = counter + 1
            elif (i == 27):
                dataOutputs.append(float(splitData[i]))
            else:
                currentSample.append(float(splitData[i]))
        dataInputs.append(currentSample)
    data.close()
    return dataInputs, dataOutputs

def loadLetters():
    letterToIntDict = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "J": 9,
            "K": 10,
            "L": 11,
            "M": 12,
            "N": 13,
            "O": 14,
            "P": 15,
            "Q": 16,
            "R": 17,
            "S": 18,
            "T": 19,
            "U": 20,
            "V": 21,
            "W": 22,
            "X": 23,
            "Y": 24,
            "Z": 25
    }
    data = open("datasets/letter-recognition.data")
    lines = data.readlines()
    dataInputs = []
    dataOutputs = []
    for line in lines:
        splitData = line.split(",")
        currentSample = []
        for i in range(len(splitData)):
            if (i == 0):
                dataOutputs.append(letterToIntDict[splitData[i]])
            else :
                currentSample.append(int(splitData[i]))
        dataInputs.append(currentSample)
    data.close()
    newInputs = []
    newOutputs = []
    noOfEachClass = [0]*26
    for i in range(len(dataInputs)):
        if (noOfEachClass[dataOutputs[i]] < 40):
            newInputs.append(dataInputs[i])
            newOutputs.append(dataOutputs[i])
            noOfEachClass[dataOutputs[i]] = noOfEachClass[dataOutputs[i]] + 1
        if (len(newInputs) == 1040):
            break
    return newInputs, newOutputs

datin, datout = loadLetters()

def loadCreditCard():
    dataInputs = []
    dataOutputs = []
    with open("datasets/default of credit card clients.csv", "r", newline='') as readCsvfile:
        reader = csv.DictReader(readCsvfile)
        init = True
        counter = 0
        for instance in reader:
            currentSample = []
            if (init):
                init = False
                continue
            for key, value in instance.items():
                if (key == "Y"):
                    dataOutputs.append(int(value))
                elif (key != ""):
                    currentSample.append(int(value))
            dataInputs.append(currentSample.copy())
            counter = counter + 1
            if (counter > 3):
                break

#di , do = loadCreditCard()
    

# Not done
# Classification and numerical
# Online handwritten assamese, lots of features, over 9000 samples
# Default of credit cards, 30000 examples, 24 features
# Mammographic mass, 6 features, 961 samples, missing values
# Breast cancer coimbra, 10 features, 116 samples, might have missing don't know
# Parkinson spiral drawings, 77 samples, 7 features
# LSVT, 309 features, 126 samples
# Gene expression, 20531 features!
# Grammatical facial experssions, 27965 samples, 100 features
# Ozone level detection, 2536 samples, 74 features, missing values
    
