import numpy as np
import csv 
import sys
import math
import random


def createLists(size):
    res = []
    for i in range(size):
        res.append(list())
    return res

def makeKNNData(data):
    KNNData = []
    KNNData = createLists(len(data[0]))
    for i in range(len(data)):
        for j in range(len(data[0])):
            val = data[i][j]
            KNNData[j].append( (i,val) )
    #print(KNNData)
    #(f'LENLEN {len(KNNData[0])}')
    return KNNData
            


def getData(dataIn):
    csvFile = open(dataIn)
    read_data = csv.reader(csvFile)
    res = []
    count = 0
    hitCount = 0
    for row in read_data:
        temp = row[2:]
        res.append(temp)
        count += 1
        #print(row[-1])
        if float(row[-1]) == 1:
            hitCount += 1
    underlyingPercent = hitCount/count
    return res, underlyingPercent
    
def distance(distances):
    res = 0
    for dis in distances:
        res += dis**2
    res = math.sqrt(res)
    return res


def updateShortests(index, currDis, dlist, k):
    temp = ( index ,currDis)
    if len(dlist) < k:
        dlist.append(temp)
    if len(dlist) >= k:
        dlist.sort(key = lambda x:x[1])
        dlist.pop()
    return dlist
    


def getPred(dlist, data, underlyingPercent):
    score = 0
    for index,dist in dlist:
        row = data[-1][index]
        label = float(row[-1])
        if label == 1:
            score += 1
        if label == 0:
            score -= 1
    #print(score, underlyingPercent)
    if score > 0:
        return 1
    if score < 0:
        return 0
    if score == 0:
        rand = random.randint(0,100)
        if rand <= underlyingPercent * 100:
            return 1
        else:
            return 0
         



def predict(line, trainData, k, underlyingPercent):
    shortests = []
    distances = []
    for i in range(len(trainData[0])):
        distances = []
        for j in range(10):
            #print(trainData[j][i])
            diff = float(line[j]) - float(trainData[j][i][1])
            #print(diff)
            if float(line[j]) != 0:
                val = diff/float(line[j])
                distances.append((diff/float(line[j])))
            if float(line[j]) == 0:
                #print(f'OTHER OTHER')
                distances.append(diff)
        fullDistance = distance(distances)
        shortests = updateShortests(i,fullDistance, shortests, k)
    predFinal = getPred(shortests, trainData, underlyingPercent)
    return predFinal
        
        

def test(testData, trainData, k, underlyingPercent):
    res = []
    labels = []
    hitCount = 0
    count = 0
    for row in testData:
        label = row[-1]
        prediction = predict(row[:-1], trainData, k, underlyingPercent)
        res.append(prediction)
        labels.append(label)
        if count % 25 == 0:
            print(count)
        count += 1
        if float(label) == float(prediction):
            hitCount += 1
    print(f'KNN acc: {hitCount/count}')
    return res, hitCount/count
        
        


def writer(preds, outfile):
	with open(outfile, 'w') as filehandle:
		for line in preds:
			filehandle.write(f'{line}\n')



def metricsWriter(testAcc, name):
	with open(name, 'w') as filehandle:
		filehandle.write(f'error(test): {testAcc}\n')



def main():
    if len(sys.argv) > 1:
        trainInput = sys.argv[1]
        testInput = sys.argv[2]
        testOutput = sys.argv[3]
        metricsOutput = sys.argv[4]
        k = int(sys.argv[5])

        dataTrain, underlyingPercent = getData(trainInput)
        dataTest, x = getData(testInput)

        KNNtrain = makeKNNData(dataTrain)
        res, testAcc = test(dataTest, KNNtrain, k, underlyingPercent)
        #print(f' IMPORTANT LEN : {len(res)}')
        writer(res, testOutput)
        metricsWriter(testAcc, metricsOutput)



if __name__ == '__main__':
    main()