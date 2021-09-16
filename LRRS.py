import numpy as np
import sys
import csv
import copy

learnRate = .1

def getData(filein):
    res = []
    csvFile = open(filein)
    read_csv = csv.reader(csvFile, delimiter = '\t')
    count = 0
    for row in read_csv:
        count += 1
        if count == 1:
            continue
        res.append(row)
    return res

def rowVector(row):
    label = row[-1]
    index = []
    for i in range(len(row) - 1):
        if int(row[i]) == 1:
            index.append(i)
    return label, index
        

def sparseDot(rowVec, params):
    res = 0.0
    # print(rowVec)
    for index in rowVec:
        res += params[index]

    ### Adding bias
    res += params[-1]
    return res

def sigmoid(val):
    return np.exp(val)/(1+np.exp(val))

def doStep(label,rowVec, params, count):
    currVal = sparseDot(rowVec, params)
    sigVal = sigmoid(currVal)
    #print(label)
    changeVal = int(label) - sigVal
    params[-1] += (learnRate/count) * changeVal
    for val in rowVec:
        params[val] = params[val] + ((learnRate*(1/ count)) * changeVal)
    return params


def SGDTrain(data, numEpochs):
    params = np.zeros(len(data[0]))

    currEpoch = 0
    while currEpoch < numEpochs:
        print(currEpoch)
        currEpoch += 1
        for row in data:
            #print(row)
            label, rowVec = rowVector(row)
            params = doStep(label, rowVec, params, len(data))
    
    print(params)
    return params

def getPredict(rowVec, params):
    currVal = sparseDot(rowVec, params) 
    sigVal = sigmoid(currVal)
    print(f'PREDICT {sigVal}')
    if (sigVal > .5):
        return 1
    if (sigVal <= .5):
        return 0   

def predict(params, data):
    zeros = 0
    ones = 0
    hits = 0
    total = len(data)
    guesses = []
    for row in data:
        label, rowVec = rowVector(row)
        label = int(label)
        prediction = getPredict(rowVec, params)
        guesses.append(prediction)
        #print(label, prediction)
        if prediction == 0:
            zeros += 1
        if prediction == 1:
            ones += 1
        if label == prediction:
            hits += 1
            
    print(guesses)
    acc = hits/total
    print(f'ACC: {acc}')
    print(f'Zeros: {zeros}, Ones {ones}')
    return guesses, acc
        
def writer(words, outfile):
	with open(outfile, 'w') as filehandle:
		for line in words:
			filehandle.write(f'{line}\n')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        trainInput = sys.argv[1]
        testInput = sys.argv[2]
        trainOut = sys.argv[3]
        testOut = sys.argv[4]
        metricsOut = sys.argv[5]
        numEpochs = int(sys.argv[6])
        learnRate = float(sys.argv[7])

        trainData = getData(trainInput)
        testData = getData(testInput)
        #print(trainData)
        params = SGDTrain(trainData, numEpochs)
        trainGuesses, trainAcc = predict(params,trainData)
        testGuesses, testAcc = predict(params,testData)

        writer(trainGuesses, trainOut)
        writer(testGuesses, testOut)



        