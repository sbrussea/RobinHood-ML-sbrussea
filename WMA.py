import numpy as np
import csv
import sys
import math

def getData(fileIn):
	tsvFile = open(fileIn)
	read_tsv = csv.reader(tsvFile, delimiter = "\t")
	rowCount = 0
	res = []
	for row in read_tsv:
		if (rowCount > 0):
			res.append(row)
		rowCount += 1
	#print(res)
	return res


def initWeights(size):
    res = np.ones(size)/size
    return res


def listErrors(row, label):
    errors = set()
    correct = set()
    for i in range(len(row)):
        val = row[i]
        if int(val) != label:
            errors.add(i)
        if int(val) == label:
            correct.add(i)
    return errors,correct
        
def getError(errorCount, length):
    return errorCount/length

def getConf(res):
    return abs(np.sin(res))

def predictGiven(data,weights):
    res = 0
    for i in range(len(data)):
        val = int(data[i])
        if val == 0:
            res += -1 * weights[i]
        if val == 1:   
            res += 1 * weights[i]
    conf = getConf(res)
    if res > 0:
        return 1,conf
    else:
        return 0,conf




def weightedMajority(trainData, B):
    weights = initWeights(len(trainData[0]) - 1)
    print(weights)
    lengthData = len(trainData[0])
    for row in trainData:
        label = int(row[-1])
        data = row[:-1]

        predict, conf = predictGiven(data, weights)
        if predict != label:
            errors, correct = listErrors(data,label)
            for val in errors:
                weights[val] = weights[val] * B 
        
    
    return weights
        

def testWMA(data, weights):
    hits = 0
    count = len(data)
    print(weights)
    for row in data:
        label = int(row[-1])
        data = row[:-1]

        predict,conf = predictGiven(data, weights)
        if predict == label:
            hits += 1
    print(f'Acc: {hits/count}')



def writer(preds, outfile):
	with open(outfile, 'w') as filehandle:
		for line in preds:
			filehandle.write(f'{line}\n')



def metricsWriter(testAcc, name):
	with open(name, 'w') as filehandle:
		filehandle.write(f'error(test): {testAcc}\n')

def main():
    dataTrain = getData('BDTtrain.tsv')
    dataTest = getData('BDTTest.tsv')
    weights = weightedMajority(dataTrain, .85)
    testWMA(dataTest, weights)

    if len(sys.argv) > 1:
        trainInput = sys.argv[1]
        testInput = sys.argv[2]
        testOutput = sys.argv[3]
        metricsOutput = sys.argv[4]

        dataTrain = getData(trainInput)
        dataTest = getData(testInput)

        #print(f' IMPORTANT LEN : {len(res)}')
        #writer(res, testOutput)
        #metricsWriter(testAcc, metricsOutput)



if __name__ == '__main__':
    main()