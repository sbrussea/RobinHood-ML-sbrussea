import numpy as np
import sys
import csv
import math
import copy
import matplotlib.pyplot as plt
import random



learnRate = 0
hiddenUnits = 1
crossEntropyTrain = list()
crossEntropyValid = list()

class netVals(object):
    def __init__(self, x, a, z, b, yhat):
        self.x = x
        self.a = a
        self.z = z
        self.b = b
        self.yhat = yhat
        self.J = None


### Feed numbers in in list form bc tuple "does not support assignment"
def create0(size):
    size[1] = size[1] + 1
    res = np.zeros(size)
    return res


### Might need to change this
## This size would NOT Include the bias 
def createRand(size):
    res = []
    for i in range(size[0]):
        temp = []
        temp.append(0)
        for i in range(size[1]):
            val = random.uniform(-0.1,0.1)
            temp.append(val)
        res.append(temp)
    
    finalRes = np.array(res)
    return finalRes

def createVector(init_flag, size):
    if init_flag == 1:
        return createRand(size)
    if init_flag == 2:
        return create0(size)

def getCrossEntropy(data,label,yhat):
    ### should create a np.array() to sum over
    logyhat = np.log(yhat)
    crossEntropy = np.array([logyhat[label]])
    return crossEntropy * -1



def NNforward(alpha, beta, data, label):
    #print(alpha)
    ### This gets a_j but might need to change alpha vector bias
    #print(data)
    temp = np.array(data)
    if checkShapes(alpha,temp):
        a_j = np.dot(alpha,temp)
    #a_j = alpha.dot(data)
    #print(a_j)

    ### Tested out in console looks good
    z_j = np.array([1])
    ### From recitation example to take out
    #z_j2 = np.tanh(a_j)
    z_j2 = (1/(1+np.exp(-a_j)))
    ### This just puts in the first index to always be a 1
    z_j = np.concatenate((z_j,z_j2))
    #print(z_j)

    ### Third step of summation over the 
    if checkShapes(beta,z_j):
        b_k = np.dot(beta,z_j)

    ### Now to get the yhat
    divider = np.exp(b_k)
    ### might not be right, could need to be down a specific line
    sumD = np.sum(divider, axis = 0)
    yhat = np.exp(b_k)/sumD
    #print(f'Yhat = {yhat}')

    ### J = 1/n sum sum yk*log(yk) ?
    # print(f'Cross Ent = {J}')
    obj = netVals(data, a_j, z_j, b_k, yhat)
    return obj

def checkShapes(a,b):
    innerA = a.shape[1]
    innerB = b.shape[0]
    if innerA != innerB:
        print('FAILURE')
    return innerA == innerB

def NNbackward(obj,data, beta,label):
    labelArray = createVector(2, [len(obj.yhat), 0])
    labelArray[label] = 1

    ## might need label +1 or label -1
    #dldyHat = label/data[0][label]
    #Rebeginning
    #dJdYhat = 

    ## Beta section
    #
    ### d(loss)/d(b)
    ### Not sure about this but it is based entirely for the tinyOutput
    #print(obj.yhat)
    #print(obj.yhat)
    dLdb = np.subtract(obj.yhat,labelArray.T)
    #dLdb = np.subtract(labelArray.T,obj.yhat)
    #print(dLdb)

    #bk = sum(Beta*z_j)
    #dldb * z? bc z is d of sum(B*z)
    ## was getting (5,) kinda weird but reshaping to (5,1) works\
    if len(obj.z.shape) == 1:
        obj.z = obj.z.reshape(len(obj.z),1)
    
    if checkShapes(dLdb.T,obj.z.T):
        dLdbeta = np.dot(dLdb.T,obj.z.T)
    #End of dldbeta

    ### Alpha section
    if checkShapes(dLdb,beta):
        dldz = np.dot(dLdb, beta)
    ### One element too long but we can just pop it

    dldz = np.array(dldz[0][1:])
    
    #changing for clarity into arrays
    dlda = np.array(obj.z * (1-obj.z))
    #print(dlda)
    #Pop off first element
    dlda = np.array(dlda[1:])
    ### It could be either one here
    ## Changing for clarity 
    dlda2 = np.array(np.multiply(dlda.T ,dldz))
    if type(data) == list:
        data = np.array(data).reshape(len(data), 1)
    if checkShapes(dlda2.T, data.T):
        dldalpha = np.dot(dlda2.T, data.T)

    return [dldalpha, dLdbeta]
    


def crossAdd(fedData,alpha,beta):
    res = 0
    for row in fedData:
        label = int(row[0])
        tempRow = copy.deepcopy(row)
        tempRow[0] = 1
        obj = NNforward(alpha,beta, tempRow, label)
        obj.J = getCrossEntropy(tempRow,label, obj.yhat)
        #print(f'Cross Ent = {obj.J}')
        res += obj.J[0]
    return res

def SGD(trainFile, validFile, numEpochs, init_flag):
    ##init parameters a and B
    tsvFile = open(trainFile)
    read_tsv = csv.reader(tsvFile, delimiter = ",")
    trainData = getData(read_tsv)
    
    tsvFile = open(validFile)
    read_tsv = csv.reader(tsvFile, delimiter = ",")
    validData = getData(read_tsv)
    countValid = 0
    for row in validData:
        countValid += 1
    ### counting params for alpha vector
    paramCount = 0
    for i in trainData[0]:
        paramCount += 1
    paramCount = paramCount - 1
    alpha = createVector(init_flag, [hiddenUnits, paramCount])
    
    ### we have 10 possible output
    beta = createVector(init_flag, [16, hiddenUnits])

    epoch = 0
    count = 0

    while epoch < numEpochs:
        epoch += 1
        print(f'Epoch = {epoch}')
        avgCrossEntropyTrain = 0
        avgCrossEntropyValid = 0
        countTrain = 0

        tsvFile = open(trainFile)
        read_tsv = csv.reader(tsvFile, delimiter = ",")
        trainData = getData(read_tsv)
        random.shuffle(trainData)
        ## kind of annoying but after it finishes all its rows after first epoch
        ## it just stops working? 
        tsvFile = open(validFile)
        read_tsv = csv.reader(tsvFile, delimiter = ",")
        validData = getData(read_tsv)
        

        for row in trainData:
            #print(row)
            countTrain += 1
            label = int(row[0])
            tempRow = copy.deepcopy(row)
            tempRow[0] = 1
            ##Compute Neural Network layers
            #First forward 
            obj = NNforward(alpha,beta, tempRow, label)
            #obj.J = getCrossEntropy(tempRow,label, obj.yhat)
            #avgCrossEntropyTrain += obj.J[0]
            # print(f'Cross Ent = {obj.J}')

            # avgCrossEntropy += obj.J[0]
            #Next Backward
            gs = NNbackward(obj, tempRow, beta, label)
            g_a = gs[0]
            g_b = gs[1]

            # Update parameters
            # a = a - g_a * learnRate
            alpha = alpha - g_a * learnRate
            # b = b - g_b * learnRate
            beta = beta - g_b * learnRate

        avgCrossEntropyTrain += crossAdd(trainData,alpha,beta)
        avgCrossEntropyValid += crossAdd(validData,alpha,beta)
      
        tempTrainEnt = avgCrossEntropyTrain/countTrain
        #print(avgCrossEntropyTrain/countTrain)
        tempValidEnt = avgCrossEntropyValid/countValid
        #print(avgCrossEntropyValid/countValid)
        crossEntropyTrain.append(tempTrainEnt)
        crossEntropyValid.append(tempValidEnt)
            
    #print(crossEntropyTrain, crossEntropyValid)
    return alpha,beta


def predict(alpha, beta, fileD):
    #print(alpha,beta)
    tsvFile = open(fileD)
    read_tsv = csv.reader(tsvFile, delimiter = ",")
    data = getData(read_tsv)
    hit = 0
    total = 0
    res = []
    for row in data:
        total += 1
        label = row[0]
        tempRow = copy.deepcopy(row)
        tempRow[0] = 1
        localMax = 0
        localMaxI = 0
        ### Just get a prediction vector
        ### forward (data, alpha, beta)
        #print(tempRow)
        temp = NNforward(alpha,beta,tempRow,label)
        for i in range(len(temp.yhat)):
            if temp.yhat[i] > localMax:
                localMax = float(temp.yhat[i])
                localMaxI = i
        # for val in temp.yhat:
        #     print(val)
        #     print(type(val))
        #     if float(val) > localMax:
        #         localMax = float(val)
        guess = localMaxI
        ### grab the most likely value
        #res.append(likelyVal)
        res.append((guess, label))

        ### compare to label
        if int(guess) == label:
            hit += 1

        # Check if hit or miss
    percentage = 1 - (hit/total)
    # print(res)
    # print(percentage)
    # print("done")
    return res, percentage

        
def writer(data, fileOut):
	with open(fileOut, 'w') as filehandle:
		for line in data:
			filehandle.write(f'{line}\n')
		

def metricsWriter(validAcc, trainAcc, dataTrain, dataValid,name):
    with open(name, 'w') as filehandle:
        for i in range(len(dataTrain)):
            filehandle.write(f'epoch={i+1} crossentropy(train): {dataTrain[i]}\n')
            filehandle.write(f'epoch={i+1} crossentropy(validation): {dataValid[i]}\n')

        filehandle.write(f'error(train): {trainAcc}\n')
        filehandle.write(f'error(validation): {validAcc}\n')




def getData(read_tsv):
    res = []
    for row in read_tsv:
        res.append(row)
    for val1 in res:
        for i in range(len(val1)):
            val1[i] = int(val1[i])

    return res



if __name__ == '__main__':

    if len(sys.argv) > 1:
        trainInput = sys.argv[1]
        validationInput = sys.argv[2]
        trainOut = sys.argv[3]
        validOut = sys.argv[4]
        metricsOut = sys.argv[5]
        numEpochs = int(sys.argv[6])
        hiddenUnits = int(sys.argv[7])
        init_flag = int(sys.argv[8])
        learnRate = float(sys.argv[9])
        #test()

        ## Train session
        alpha, beta = SGD(trainInput, validationInput, numEpochs, init_flag)
        #print(f'Alpha\n{alpha}')
        #print(f'Beta\n{beta}')
        ## Predict session
        trainRes, errorTrain = predict(alpha, beta, trainInput)
        validRes, errorValid = predict(alpha, beta, validationInput)
        print(errorTrain, errorValid)
        

        writer(trainRes, trainOut)
        writer(validRes, validOut)

        metricsWriter(errorValid, errorTrain, crossEntropyTrain, crossEntropyValid, metricsOut)
        