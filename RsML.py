import csv
import os
import time
import random
import scipy.stats as st
import numpy as np
import robin_stocks.robinhood as rs

import dataMake as dm
import BDT as BDT
from BDT import *
import matplotlib.pyplot as plt

BDTmaxDepth = 5
TotalparamCount = 0
dateDiffs = [1,2,3,4,5,7,10,14,30]



def extractTickers(dataUse):
    res = []
    for row in dataUse:
        ticker = row.pop()
        res.append(ticker)
    return dataUse, res

def getAllData(forwardDate, changeAmount):
    
    dataUse = dm.getData(['NKE'], dateDiffs, forwardDate, changeAmount)
    #dataUse = dataUse[300:]
    print(f'Length of data: {len(dataUse)}')
    #print(dataUse[10:15])
    dataUse, tickers = extractTickers(dataUse)
    Cats = dm.makeDataCats(dateDiffs)
    #print(dm.intoDF(data))
    return dataUse, Cats, tickers


def binarize(dataUse):
    #print(dataUse)
    res = []
    for row in dataUse:
        temp = []
        realData = row[2:]
        for val in realData:
            val = float(val)
            if val > 0:
                temp.append(1)
            else:
                temp.append(0)
        res.append(temp)
    return res
        

def splitData(data, dataNN, forwardDate):
    print(len(data), len(dataNN))
    trainData = []
    testData = []
    data = data[:-forwardDate]
    trainDataNN = []
    testDataNN = []
    index = 0
    testLabels = []
    for row in data:
        rand = random.randint(0,100)
        if rand < 25:
            testDataNN.append(dataNN[index])
            testLabels.append(data[index][-1])
            testData.append(row)
        else:
            trainDataNN.append(dataNN[index])
            trainData.append(row)
        
        index += 1
    #print(len(trainData))
    #print(len(testData))
    print(f'LenTestLabs = {len(testLabels)}')
    print(f'LenTestData = {len(testData)}')
    return trainData, testData, trainDataNN, testDataNN, testLabels

def splitDataNN(data, labels):
    trainData = []
    testData = []
    trainLab = []
    testLab = []
    index = 0
    for row in data:
        rand = random.randint(0,100)
        if rand < 30:
            testData.append(row)
            testLab.append(labels[index])
        else:
            trainData.append(row)
            trainLab.append(labels[index])
        index += 1
    #print(len(trainData))
    #print(len(testData))
    return trainData, testData, trainLab, testLab


def getAttrs(dataCats):
    attrs = []
    for key in dataCats:
        if dataCats[key] > 1:
            attrs.append(key)
    #print(attrs)
    return attrs

def createBDTCSV(attrs, trainData, testData):
    attrs.append('Label')
    print(len(attrs))
    with open('BDTtrain.tsv', 'w') as tsvFile:
        tsvWriter = csv.writer(tsvFile, delimiter = '\t')
        tsvWriter.writerow(attrs)
        tsvWriter.writerows(trainData)
    with open('BDTtest.tsv', 'w') as tsvFile:
        tsvWriter = csv.writer(tsvFile, delimiter = '\t')
        tsvWriter.writerow(attrs)
        tsvWriter.writerows(testData)

def getForwardNumberDiff(data, forwardDate):
    count = len(data)
    index = 0
    forwardL = []
    for row in data:
        if (index + forwardDate) >= count:
            continue
        else:
            forwardChange =  float(data[index + forwardDate][0]) - float(row[0])
            forwardL.append(forwardChange)
        index += 1
    return forwardL

def getNumVal(z):
    if z > 0:
        #print(st.norm.sf(abs(z)))
        val = st.norm.sf(abs(z))
        res = 16 - (16 * val)
        return res
    if z <= 0:
        val = st.norm.sf(abs(z))
        res = 16 * val
        return res

def adjustNNNums(forwardNumsDiff, changeAmount):
    #print(len(forwardNumsDiff))
    relativeChangeVals = []
    for val in forwardNumsDiff:
        relativeChange = val - changeAmount
        relativeChangeVals.append(relativeChange)
    relChangeVals = np.asarray(relativeChangeVals, dtype= np.float64)
    mean = relChangeVals.mean()
    std = relChangeVals.std()
    adjustedVals = []
    for val in relChangeVals:
        zScore = (val - mean) / std
        temp = getNumVal(zScore)
        adjustedVals.append(int(temp))
    #adjustedVals = np.asarray(adjustedVals, dtype = np.float64)
    return adjustedVals
    
        
    


def createNNData(data, forwardDate, changeAmount):
    forwardNumsDiff = getForwardNumberDiff(data, forwardDate)
    adjustedNums = adjustNNNums(forwardNumsDiff, changeAmount)
    NNData = []
    temp = copy.deepcopy(data)
    labs = []
    print(f'HERE {len(data)}')
    for row in temp:
        label = row[-1]
        if label == 0 or label == 1:
            NNData.append(row[:-1])
            labs.append(row[-1])
        #else:
           # NNData.append(row[:-1])
    NNfinal = binarize(NNData)
    for label, row in zip(adjustedNums, NNfinal):
        row.insert(0,label)
    return NNfinal, labs

def createNNCSV(trainData, testData):
    with open('NNtrain.csv', 'w') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(trainData)
    with open('NNtest.csv', 'w') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(testData)


def gnbVal(val):
    return np.exp(val)/ (1+np.exp(val))

def returnGNBData(data):
    res = []
    attrData = []
    tempData = copy.deepcopy(data)
    count = 0
    for row in tempData:
        if count > max(dateDiffs):
            attrData.append(row[2:-1])
        count += 1
        
    attrData = np.asarray(attrData, dtype = np.float64)
    columnMeans = attrData.mean(axis = 0)
    columnStd = attrData.std(axis = 0)
    # print(attrData)
    # ### cols
    # print(tempData[0])
    for i in range(len(tempData)):
        tempList = []
        for j in range(2,len(tempData[0])-1):
            #print(tempData[4][j])
            val = tempData[i][j]
            val = float(val)
            zScore = (val - columnMeans[j-2]) / columnStd[j-2]
            tempVal = gnbVal(zScore)
            tempList.append(tempVal)
        res.append(tempList)

    for i in range(len(res)):
        res[i].append(data[i][-1])
    print(f'Len{len(res)}')
    return res

        

def createGNBData(trainData, testData):
    trainDataGNB = returnGNBData(trainData)
    testDataGNB = returnGNBData(testData)
    return trainDataGNB, testDataGNB

def createGNBCSV(trainData, testData):
    with open('GNBTrain.csv', 'w') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(trainData)
    with open('GNBTest.csv', 'w') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(testData)

def shuffleLists(data):
    random.shuffle(data)
    return data


def commands():
    global BDTCommand
    BDTCommand = f'python BDT.py BDTtrain.tsv BDTtest.tsv {BDTmaxDepth} BDTtrain.labels BDTtest.labels BDTmetrics.txt'

    global NNCommand
    NNCommand = 'python3 NN.py NNtrain.csv NNtest.csv \
NNtrainLabs.labels NNtestLabs.labels NNmetrics.txt \
60 5 1 0.1'

    global GNBCommand 
    GNBCommand = f'python3 GNB.py GNBTrain.csv GNBTest.csv GNBTrainL.labels GNBTestL.labels GNBMetrics.txt {TotalparamCount-2}'
    #return BDTCommand
    global KNNCommand
    KNNCommand = f'python3 KNN.py KNNTrain.csv KNNTest.csv KNNTestL.labels KNNMetrics.txt {K}'

    global LRCommand
    LRCommand = f'python3 LRRS.py BDTtrain.tsv BDTtest.tsv LRTrain.labels LRTest.labels LRMetrics.txt 100 .1'

def testNN():
    NNTrainResults = []
    trainFile = open('NNtrainLabs.labels')
    trainLines = trainFile.readlines()
    hitsTrain = 0
    missesTrain = 0 
    for line in trainLines:
        nums = line.split(', ')
        predict = int(nums[0][1:]) - 8
        real = int(nums[1][:-2]) - 8
        if np.sign(predict) == np.sign(real):
            hitsTrain += 1
        if np.sign(predict) != np.sign(real):
            missesTrain += 1

    testFile = open('NNtestLabs.labels')
    testLines = testFile.readlines()
    NNTestResults = []
    hitsTest = 0
    missesTest = 0 
    for line in testLines:
        nums = line.split(', ')
        predict = int(nums[0][1:]) - 8
        real = int(nums[1][:-2]) - 8
        if predict >= 0:
            NNTestResults.append(1)
        if predict < 0:
            NNTestResults.append(0)
        if np.sign(predict) == np.sign(real):
            hitsTest += 1
        if np.sign(predict) != np.sign(real):
            missesTest += 1
    print(hitsTest, missesTest)
    return NNTestResults

def testGNB():
    testRes = []
    testFile = open('GNBTestL.labels')
    testLines = testFile.readlines()
    for line in testLines:
        testRes.append(int(line))

    trainRes = []
    trainFile = open('GNBTrainL.labels')
    trainLines = trainFile.readlines()
    for line in trainLines:
        trainRes.append(int(line))

    return testRes

def createKNNCSV(trainData, testData):
    with open('KNNTrain.csv', 'w') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(trainData)
    with open('KNNTest.csv', 'w') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(testData)

def writer(preds, outfile):
	with open(outfile, 'w') as filehandle:
		for line in preds:
			filehandle.write(f'{line}\n')

def readandReturn(file):
    res = []
    toRead = open(file)
    lines = toRead.readlines()
    for line in lines:
        res.append(int(line))
    return res


def checkMistake(val, lab):
    ### Mistake is true here
    if val > 0 and lab > 0:
        return False
    if val > 0 and lab == 0:
        return True
    if val <= 0 and lab == 0:
        return False
    if val <= 0 and lab == 1:
        return True
    

def getEnsembleVal(val):
    if val > 0:
        return 1
    if val <= 0 :
        return 0

def ensemble(NNResults, testLabels):
    ### This will be the adaboost algo that was used in my ML class and slightly changed
    BDTResults = readandReturn('BDTtest.labels')
    GNBResults = readandReturn('GNBTestL.labels')
    KNNResults = readandReturn('KNNTestL.labels')
    LRResults = readandReturn('LRTest.labels')
    #### We want to remove the today tag that we put on it
    #testLabels.pop()
    # print(f'BDT {len(BDTResults)}')
    # print(f'NN {len(NNResults)}')
    # print(f'GNB {len(GNBResults)}')
    testSetLength = len(BDTResults)
    classifiers = [BDTResults, NNResults, GNBResults, KNNResults]
    initWeight = 1/len(classifiers)
    weights = []
    for i in range(len(classifiers)):
        weights.append(initWeight)
    res = []
    #for i in range(100):
     #   print(GNBResults[i], BDTResults[i])
    print(len(GNBResults), testSetLength)
    #print(len(testLabels), len(NNResults))
    i = 0
    while i < testSetLength:
        val = 0
        for classifier,weight in zip(classifiers,weights):
            tempVal = classifier[i] 
            if tempVal == 0:
                val += -1 * weight
            if tempVal == 1:
                val += 1 * weight
        if checkMistake(val, testLabels[i]):
            j = 0
            for classifier,weight in zip(classifiers,weights):
                tempVal = classifier[i]
                if checkMistake(tempVal, testLabels[i]):
                    weights[j] *= .70
                j += 1
        i += 1
        
    hits = 0
    accuracy1 = 0
    total1 = 0
    accuracy0 = 0
    total0 = 0
    for i in range(testSetLength):
        val = 0
        for classifier,weight in zip(classifiers,weights):
            tempVal = classifier[i] 
            if tempVal == 0:
                val += -1 * weight
            if tempVal == 1:
                val += 1 * weight

        predict = getEnsembleVal(val)
        res.append(predict)

        if predict == testLabels[i]:
            hits += 1
            if predict == 1:
                accuracy1 += 1
            if predict == 0:
                accuracy0 += 1
        if testLabels[i] == 0:
            total0 += 1
        if testLabels[i] == 1:
            total1 += 1

    print(weights)
    print(f'Above Accuracy = {accuracy1/total1}, totalAbove: {accuracy1}/{total1}')
    print(f'Below Accuracy = {accuracy0/total0}, Below Actual: {accuracy0}/{total0}')
    print(f'Ensemble Accuracy: {hits/testSetLength}')
    writer(res, 'ensembleLabels.txt')
    return hits/testSetLength
        

def main():
    global K
    K = 5
    forwardDate = 8
    changeAmount= 2
    commands()
    dataUse, dataCats, tickers = getAllData(forwardDate, changeAmount)
    todayData = copy.deepcopy(dataUse[-1])
    dataNN, labs = createNNData(dataUse, forwardDate, changeAmount)
    
    trainData, testData, trainDataNN, testDataNN, testLabels = splitData(dataUse, dataNN, forwardDate)
    
    trainDataBin = binarize(trainData)
    testDataBin = binarize(testData)
    trainDataBin = shuffleLists(trainDataBin)
    attrsBDT = getAttrs(dataCats)

    ### BDT:
    print(f'BDT Beginning')
    createBDTCSV(attrsBDT, trainDataBin, testDataBin)
    os.system(BDTCommand)
    print(f'BDT Done')

    # print(testDataBin[15], testDataNN[15])
    # print(len(todayData[2:]), (todayData[2:]))


    ### NN:
    print(f'NN Beginning')
    ### NN Command
    ### Need to uncomment this but for testing now it is fine
    ### Tested and not a big difference between shuffling and not shuffling initially
    #trainDataNN = shuffleLists(trainDataNN)
    createNNCSV(trainDataNN, testDataNN)
    os.system(NNCommand)
    NNTestResults = testNN()
    print(f'NN Done')
    


    ### GNB:
    print(f'GNB Beginning')
    trainDataGNB, testDataGNB = createGNBData(trainData, testData)
    global TotalparamCount
    TotalparamCount = len(trainData[0]) - 1
    commands()
    createGNBCSV(trainDataGNB, testDataGNB)
    os.system(GNBCommand)
    print(f'GNB Done')
    #GNBTestResults = testGNB()

    
    
    ### KNN
    print(f'KNN Beginning')
    createKNNCSV(trainData, testData)
    os.system(KNNCommand)
    print(f'KNN Done')

    ### LR
    #print(f'LR Beginnning')
    #os.system(LRCommand)
    #print(f'LR Done')


    ### Only with 4 classifiers right now
    return ensemble(NNTestResults, testLabels)



if __name__ == '__main__':
    print(main())
    


