import sys
import numpy as np
import csv
from scipy import *
import copy

labelsList = []
bigattrList = []
attrDict = dict()
maxDepth = 0

class Node(object):
	def __init__(self,currList, depth):
		self.right = None
		self.left = None
		self.allList = currList
		self.attrList = None
		self.seenAttrs = []
		self.rightLabel = None
		self.leftLabel = None
		self.depth = depth
		self.attributeName = None
		self.splitNum = None
		self.parent = None
		self.leftSplit = None
		self.rightSplit = None
		self.leftSplitVal = None
		self.rightSplitVal = None
	
	def __eq__(self,other):
		pass
		#return  self.seenAttrs == other.seenAttrs

#### This function gets the entropy of the currList given, the given 
### attr list should change based on the attributes given to it
def baseEntropy(node):
	labelsDict = dict()
	totalCount = 0
	res = 0
	for row in node.allList:
		label = row[-1]
		if label in labelsDict:
			labelsDict[label] += 1
		if label not in labelsDict:
			labelsDict[label] = 1
			labelsList.append(label)
		totalCount += 1
	#print(totalCount)
	for label in labelsDict:
		res += getEntropy(labelsDict[label], totalCount)
		
	res = abs(res)
	#labelsList.sort()
	return res



def getEntropy(count, total):
	frac = count / total
	res = frac * np.log2(frac)
	return res

def placeAttrs(trainFile):
	tsvFile = open(trainFile)
	read_tsv = csv.reader(tsvFile, delimiter = "\t")
	currCol = 0
	currattrList = []
	
	for row in read_tsv:
		for col in row:
			currattrList.append(col)
			#attrDict[currCol] = col
			attrDict[col] = currCol
			currCol += 1
		break
	currattrList.pop()
	return currattrList

### This function takes in the attr that is left, then breaks the attr into two
### lists in the binary form, we then return those two lists of rows in a dict
def attrSplitBin(node, attr):
	seenDict = dict()
	splitList = []
	attrCol = attrDict[attr]
	for row in node.allList:
		curr = row[attrCol]
		if (curr in seenDict):
			seenDict[curr].append(row)
		if (curr not in seenDict):
			seenDict[curr] = [row]
	for val in seenDict:
		splitList.append(seenDict[val])
	return splitList


### This function is basically doing the base Labels function but 
### gives a dif output
### code basis found in decison stump
def getSplitEntropy(splitAttr):
	splitDict = dict()
	totalCount = 0
	res = 0
	for row in splitAttr:
		label = row[-1]
		if label in splitDict:
			splitDict[label] += 1
		if label not in splitDict:
			splitDict[label] = 1
		totalCount += 1

	for label in splitDict:
		res += getEntropy(splitDict[label], totalCount)
	res = abs(res)
	return res,totalCount


### This function gets the info gained from each attr given to it
def getInfo(node, attr, baseEntropy):
	totalCount0 = 0
	totalCount1 = 0
	split0 = 0 
	split1 = 0
	#### First we split attr in binary form
	attrSplit = attrSplitBin(node, attr)
	split0,totalCount0 = getSplitEntropy(attrSplit[0])
	# print(f"ATTR SPLIT0 {attrSplit[0]}\n")
	# print(f"ATTR SPLIT1 {attrSplit[1]}\n\n\n")
	if len(attrSplit) > 1:
		split1,totalCount1 = getSplitEntropy(attrSplit[1])
	totalFinal = totalCount0 + totalCount1
	infoEntropy = (split0 * (totalCount0/totalFinal)) + (split1 * (totalCount1/totalFinal))
	#print(baseEntropy - infoEntropy)
	infoVal = (baseEntropy - infoEntropy)
	return [attrSplit, infoVal]

### from decision stump
def getMajority(listCount):
	res = dict()
	largeCount = 0
	curr = ""
	curr2 = ""
	for elem in listCount:
		#print(elem)
		val = elem[-1]
		if val in res:
			res[val] += 1
		if val not in res:
			res[val] = 1

	for val in res:
		if (curr == ""): 
			curr = val
		else: 
			curr2 = val
	### going to need to adjust this for things coming last in lex order
	if (curr2 == ""):
		return (curr, 0)
	if (res[curr] == res[curr2]):
		retVal = (curr, curr2)
		retVal = sorted(retVal)
		retValRev = []
		retValRev.append(retVal[1])
		retValRev.append(retVal[0])
		#print(f' \n\n\n\n TIE :: {retValRev}\n\n')
		return retValRev
		#retVal.sort(reverse = True)

	if (res[curr] > res[curr2]):
		return (curr, curr2)
	if (res[curr] < res[curr2]):
		return (curr2, curr)
		
			


### all from decision stump
def setLabels(node):
	res = getMajority(node.leftSplit)[0]
	res1 = getMajority(node.rightSplit)[0]
	#print(res, res1)
	node.leftLabel = res
	node.rightLabel = res1
	#print(f'LeftLabel = {node.leftLabel}\n')
	#print(f'RightLabel = {node.rightLabel}\n')
	

def prettyPrint(node, direct):
	depthShow = '| ' * node.depth
	prettyDict = dict()
	labelList = []
	currList = []
	if (direct == "left"):
		currList = node.leftSplit
	if (direct == "right"):
		currList = node.rightSplit
	if (direct == "start"):
		currList = node.allList
	for row in currList:
		label = row[-1]
		if label in prettyDict:
			prettyDict[label] += 1
		if label not in prettyDict:
			prettyDict[label] = 1
	for elem in prettyDict:
		temp = f'{prettyDict[elem]} {elem}'
		labelList.append(temp)
	if direct == "start":
		print('sbrussea')
		print(labelList)
	if direct == "left":
		print(f'{depthShow} {node.attributeName} = {node.leftSplitVal}: {labelList}')
	if direct == "right":
		print(f'{depthShow} {node.attributeName} = {node.rightSplitVal}: {labelList}')



#Check ”stopping criteria.” One being the node depth reaching the max depth (What
#other stopping criteria is there?). What happens when we reach the stopping criteria?
#• Calculate entropy and mutual information for the non-used attributes and select
#the best attribute to split
#• Split the data based on the selected attributes


def createRecursiveTree(node):
	if (node.depth > maxDepth):
		### stop the program
		### first we put in the labels of the splits, we can do this by getting 
		### majority vote like we did in decision stump
		### Lots of code so far has been from decision stump I hope thats okay
		setLabels(node.parent)
		
		#print(f'\n\nMax Depth Parent is {node.parent.attributeName} \n')
		return 
	
	

	if (node.depth <= maxDepth):
		#print(f"currNode seenAttrs {node.seenAttrs} {node.depth}")
		infoMax = 0
		attrName = "TEST ATTR NAME"
		attrSplitNum = 0
		split0 = 0
		split1 = 0

		### This will give us base Entropy at the currentNode
		baseLabEntropy = baseEntropy(node)

	### Check to see if we have hit max depth yet
	### base case 1
	### Next, we should go through each column to see what has the highest
	### information gained.
	#• Calculate entropy and mutual information for the non-used attributes and select
	#the best attribute to split
		for attr in bigattrList:
			if (attr not in node.seenAttrs):
				infoList = getInfo(node, attr, baseLabEntropy)
				infoVal = infoList[1]
				#print(attr, infoVal)
			#infoVal = infoList[1]
			#print(split0)
			#	• It is possible for attributes to have equal values for mutual information. In this case, you should split
			#on the first attribute to break ties.
				if infoVal > infoMax and (attr not in node.seenAttrs):
					attrName = attr
					attrSplitNum = attrDict[attr]
					infoMax = infoVal
					split0 = infoList[0][0]
					split1 = infoList[0][1]
		### After we jsut selected the best split, we can now move to the creation of
		### next node or ending the tree
		#• As a stopping rule, only split on an attribute if the mutual information is > 0.
		#node.attrList.remove(attrName)
		node.seenAttrs.append(attrName)
		node.attributeName = attrName
		
		if (split0 != 0):
			node.leftSplitVal = split0[0][attrSplitNum]
			node.leftSplit = split0
			#print(split1)
			#print(attrSplitNum)
			node.rightSplitVal = split1[0][attrSplitNum]
			node.rightSplit = split1
		### base case 2
		#print(infoMax, attrName, attrDict[attrName], node.attrList)
		
		if (infoMax > 0):
			
			if (infoMax <= 1):
				leftNode = Node(split0, node.depth + 1)
				rightNode = Node(split1, node.depth + 1)
				leftNode.seenAttrs = copy.deepcopy(node.seenAttrs)
				rightNode.seenAttrs = copy.deepcopy(node.seenAttrs)
				leftNode.parent = node
				rightNode.parent = node
				#print(f"Right List {rightNode.attrList}")
				#print(f"Left List {leftNode.attrList}")
				# print(f'Curr attr is : {node.attributeName}\n')
				# print(f'Curr Depth is : {node.depth}\n')
				if node.parent != None:
					#print(f'Curr Node Parent : {node.parent.attributeName}\n')
					pass
				# print(f'Left Split Val = {node.leftSplitVal}')
				# print(f"Left Split = {split0}\n\n")
				# print(f'Right Split Val = {node.rightSplitVal}')
				# print(f"Right Split = {split1}\n\n")
				
				prettyPrint(node, "left")
				node.left = createRecursiveTree(leftNode)
				prettyPrint(node, "right")
				
				node.right = createRecursiveTree(rightNode)
				
				
				
				return node

			if (infoMax > 1):
				print("Problem with infoMax line 159")
		if (infoMax <= 0):
			#print(infoMax)
			#print("\n HIT 0's\n")
			setLabels(node.parent)
			return 


def tester(root, row):
	#print(row)
	#print(root.attributeName)
	#print(root.leftSplitVal)
	
	currNode = root
	realVal = row[-1]
	testVal = None
	while (True):
		#print(currNode)
		#print(currNode.attributeName)
		#First check what direction we would be going
		splitNum = attrDict[currNode.attributeName]
		splitValRow = row[splitNum]
		leftDir = currNode.leftSplitVal
		rightDir = currNode.rightSplitVal
		### Check to see if we should be going left or right
		if (splitValRow == leftDir):
			### Check if we have a new Node to go to or if we should have label 
			if (currNode.left == None):
				testVal = currNode.leftLabel
				#print(f'PREDICTION : {testVal}\n')
				break
			if (currNode.left != None):
				#print(f'1')
				currNode = currNode.left
				continue
		if (splitValRow == rightDir):
			if (currNode.right == None):
				testVal = currNode.rightLabel
				#print(f'PREDICTION : {testVal}\n')
				break
			if (currNode.right != None):
				#print(f'2')
				currNode = currNode.right
				continue
	
	return testVal
	# if (testVal == realVal):
	# 	print("GLORY\n")
	# 	return 1
	# print("MISS\n")
	# return 0

### decision stump
def writer(L , name):
	with open(name, 'w') as filehandle:
		for line in L:
			filehandle.write(f'{line}\n')


### From decision stump again
def testTree(root, testList, testOut):
	testHits = 0
	testTotal = 0

	# tsvFile = open(testInput)
	# read_tsv = csv.reader(tsvFile, delimiter = "\t")
	res = []
	count = 0
	for row in testList:
		count += 1
		testTotal += 1
		realVal = row[-1]
		#print(f'Count: {count}')
		testVal = tester(root, row)
		if (testVal == realVal):
			testHits += 1
		res.append(testVal)
	
	writer(res, testOut)

	testAcc = (1- (testHits/testTotal))

	#print(testTotal - testHits)
	#print(1- (testHits/testTotal))
	return testAcc
		
### grabs a file and returns a list of data
def getList(infile):
	tsvFile = open(infile)
	read_tsv = csv.reader(tsvFile, delimiter = "\t")
	rowCount = 0
	res = []
	for row in read_tsv:
		if (rowCount > 0):
			res.append(row)
		rowCount += 1
	#print(res)
	return res


### From decision stump
def metricsWriter(testAcc, trainAcc, name):
	with open(name, 'w') as filehandle:
		filehandle.write(f'error(train): {trainAcc}\n')
		filehandle.write(f'error(test): {testAcc}\n')


#### Take basically from inspection
def simpleMajorityTrain(trainList, simpleOut):
	hits = 0
	total = 0
	maj = getMajority(trainList)[0]
	res = []
	for row in trainList:
		total += 1
		if row[-1] == maj:
			hits += 1
		res.append(maj)
	writer(res, simpleOut)
	trainError = 1 - (hits/total)
	return (trainError , maj)


def simpleMajorityTest(testList, testOut, maj):
	hits = 0
	total = 0
	res = []
	for row in testList:
		total += 1
		if row[-1] == maj:
			hits += 1
		res.append(maj)
	writer(res,testOut)
	testError = 1 - (hits/total)
	return testError

def BDTStart():
    trainInput = 'BDTtrain.tsv'
    testInput = 'BDTtest.tsv'
    maxDepth = 4
    trainList = getList(trainInput)
    testList = getList(testInput)
    trainOutput = 'trainOut.txt'
    testOutput = 'testOut.txt'
    metricsOutput = 'metricsOut.txt'
    trainList = getList(trainInput)
    testList = getList(testInput)
    if (maxDepth == 0):
        trainError, maj = simpleMajorityTrain(trainList, trainOutput)
        testError = simpleMajorityTest(testList, testOutput, maj)
        print(f'TestError = :{testError}')
        print(f'TrainError = :{trainError}')
        metricsWriter(testError, trainError, metricsOutput)
        
        ### just to majority vote
    #print(attrList)
    #print(attrDict)
    if (maxDepth != 0):
        root = Node(None, None)
        root.attrList = placeAttrs(trainInput)
        bigattrList = placeAttrs(trainInput)
        root.depth = 1
        
        if (maxDepth > len(bigattrList)):
            maxDepth = len(bigattrList)
            print(maxDepth)

        root.allList = trainList
        prettyPrint(root, "start")
        createRecursiveTree(root)
        
        #print(f'\n\n\n\n\n Root ATTR {root.attributeName}' )
        testAcc = testTree(root, testList, testOutput)
        trainAcc = testTree(root, trainList, trainOutput)
        metricsWriter(testAcc, trainAcc, metricsOutput)
        print(f'TestError = :{testAcc}')
        print(f'TrainError = :{trainAcc}')
    



### initial loop
if __name__ == '__main__':
	if (len(sys.argv) > 1):
		#print(sys.argv)
		trainInput = sys.argv[1]
		testInput = sys.argv[2]
		maxDepth = int(sys.argv[3])
		trainOutput = sys.argv[4]
		testOutput = sys.argv[5]
		metricsOutput = sys.argv[6]
		trainList = getList(trainInput)
		testList = getList(testInput)
		if (maxDepth == 0):
			trainError, maj = simpleMajorityTrain(trainList, trainOutput)
			testError = simpleMajorityTest(testList, testOutput, maj)
			print(f'TestError = :{testError}')
			print(f'TrainError = :{trainError}')
			metricsWriter(testError, trainError, metricsOutput)
			
			### just to majority vote
		#print(attrList)
		#print(attrDict)
		if (maxDepth != 0):
			root = Node(None, None)
			root.attrList = placeAttrs(trainInput)
			bigattrList = placeAttrs(trainInput)
			root.depth = 1
			
			if (maxDepth > len(bigattrList)):
				maxDepth = len(bigattrList)
				print(maxDepth)

			root.allList = trainList
			prettyPrint(root, "start")
			createRecursiveTree(root)
			
			#print(f'\n\n\n\n\n Root ATTR {root.attributeName}' )
			testAcc = testTree(root, testList, testOutput)
			trainAcc = testTree(root, trainList, trainOutput)
			metricsWriter(testAcc, trainAcc, metricsOutput)
			print(f'TestError = :{testAcc}')
			print(f'TrainError = :{trainAcc}')