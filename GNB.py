import numpy as np
import csv
import sys
import math


def getProb(x, mean, sd):
    leftA = math.sqrt(2 * math.pi)
    left =  1 / (leftA * sd)
    # print(f'L = {left}')

    # print(x)
    # print(mean)
    # print(sd)
    # print(f'Mayb {-.5 * (((x-mean)/ sd)**2)}')
    right = np.exp(-.5 * (((x-mean)/ sd)**2))
    # print(f'R = {right}')
    return left * right

def theirProb(x,mean, sd):
    leftA = math.sqrt(2*math.pi*(sd**2))
    leftA = 1/leftA

    rightA = np.exp((-(x-mean)**2) /(2*sd**2))

    return leftA * rightA



def getData(dataIn):
    tsvFile = open(dataIn)
    read_data = csv.reader(tsvFile)
    res = []
    count = 0
    for row in read_data:
        temp = row
        res.append(temp)
        #print(f'{count + 1} :{row[-1]}')
        count += 1
    return res


def createVector(num_voxels):
    res = np.zeros(num_voxels)
    return res


def createEsts(data, label):
    
    res = []
    means = []
    for col in range(len(data[0])-1):
        temp = 0
        count = 0
        for row in range(len(data)):
            if data[row][-1] == label:
                count += 1
                temp += float(data[row][col])

        
        means.append(temp/count)
    #print(means)
    sds= []
    for col in range(len(data[0])-1):
        temp = 0
        count = 0
        for row in range(len(data)):
            if data[row][-1] == label:
                count += 1
                temp += (float(data[row][col]) - means[col]) ** 2
        temp = temp/count
        temp = math.sqrt(temp)
        sds.append(temp)
    
    res.append(means)
    res.append(sds)
    #print(sds)
    # print(label)
    # print(res[0][0])
    return res

def getPredict(row,currDict,totalCount, partCount, num_voxels):
    #print(num_voxels)
    chance = 0
    # count = 0
    for col in range(num_voxels):
        # count += 1
        # print(f'COL : {col}')
        # print(f'Lab = {row[-1]}')
        # print(f'Pots+ {float(row[col])}')
        # print(f'Pots2 {currDict[0][col]}')
        # print(f'Pots3 {currDict[1][col]}')
        # print(f'FInal: {getProb(float(row[col]),currDict[0][col], currDict[1][col])}')

        val = np.log(getProb(float(row[col]),currDict[0][col], currDict[1][col]))
        # if val <=-1000000000:
        #     print(val)
        #     exit()
        
        # print(val)

        chance += np.log(getProb(float(row[col]),currDict[0][col], currDict[1][col]))

    chance += np.log(partCount/totalCount)
    return chance



def predict(labs,data, vox):
    res = []
    totalCount = 0
    labCount = dict()
    for row in data:
        totalCount += 1
        if row[-1] in labCount:
            labCount[row[-1]] += 1
        if row[-1] not in labCount:
            labCount[row[-1]] = 1



    for row in data:
        bestOdds = -100000000
        bestLab = ""
        #print(row)
        for label in labs:
            #print(label)
            odds = getPredict(row,labs[label], len(data), labCount[label], vox)
            # print(odds)
            #print(odds)
            if odds > bestOdds:
                bestOdds = odds
                bestLab = label
        # print(bestLab)
        res.append(bestLab)
        
    return res


def selectFeats(labsD, num_voxels, total):
    means = list()
    kickList = list()
    diffList = list()
    diff = total - num_voxels
    for lab in labsD:
        means.append(labsD[lab][0])
    print(means)
    for i in range(len(means[0])):
        diffList.append(abs(means[0][i] - means[1][i]))
    
    for i in range(diff):
        # print(min(diff))
        rem = diffList.index(min(diffList))
        diffList.pop(rem)
        means[0].pop(rem)
        
        means[1].pop(rem)

    
    

    
    
def main(num_voxels,data):
    toolsParam = createVector(num_voxels)

    labsD = dict()
    for row in data:
        if row[-1] not in labsD:
            labsD[row[-1]] = 0

    for lab in labsD:
        labsD[lab] = createEsts(data,lab)
    

    ### Add in a function to take the best num_voxel params and kick out the worst
    ### Have that new vector of ests be used in the calculation   
    #res = selectFeats(labsD, num_voxels, len(data)) 
    
    # print(labsD["No"][0])
    # print(labsD["No"][1])
    #print(labsD)
    return labsD
    

def writer(preds, outfile):
	with open(outfile, 'w') as filehandle:
		for line in preds:
			filehandle.write(f'{line}\n')

    
def getError(preds, data):
    hits = 0
    total = 0
    for i in range(len(preds)):
        total += 1
        if preds[i] == data[i][-1]:
            hits += 1
    return 1- (hits/total)
        
def metricsWriter(testAcc, trainAcc, name):
	with open(name, 'w') as filehandle:
		filehandle.write(f'error(train): {trainAcc}\n')
		filehandle.write(f'error(test): {testAcc}\n')


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        trainInput = sys.argv[1]
        testInput = sys.argv[2]
        trainOutput = sys.argv[3]
        testOutput = sys.argv[4]
        metricsOutput = sys.argv[5]
        num_voxels = int(sys.argv[6])
        
        trainData = getData(trainInput)
        testData = getData(testInput)

        labsD = main(num_voxels, trainData)

        trainPreds = predict(labsD,trainData, num_voxels)
        
        testPreds = predict(labsD,testData, num_voxels)

        trainError = getError(trainPreds, trainData)
        testError = getError(testPreds, testData)

        #print(testPreds)
        #print(trainPreds)
        writer(trainPreds, trainOutput)
        writer(testPreds, testOutput)
        metricsWriter(testError,trainError,metricsOutput)
