import csv
import robin_stocks.robinhood as rs

import numpy as np
import pandas as pd

login = rs.login("Put email here",
                "Put Password Here", 
                expiresIn = 86400)


### Order of data
dataCats = dict()
dataCats['open_price'] = 0
dataCats['close_price'] = 1
dataCats['volume'] = 2
dataCats['diff'] = 3
dataCats['perc_change'] = 4


def diffPerc(openPrice, closePrice):
    diff = closePrice - openPrice
    percChange = diff/openPrice
    return percChange, diff

def getDataFromSum(summ):
    #print(summ)
    res = list()
    openPrice = float(summ['open_price'])
    closePrice = float(summ['close_price'])
    percChange,diff = diffPerc(openPrice, closePrice)
    res.append(summ['open_price'])
    res.append(summ['close_price'])
    res.append(summ['volume'])
    res.append(diff)
    res.append(percChange)

    return res


### Will return percent change and the actual value change
def dateDifference(data, dateDiff, index):
    
    if dateDiff >= index:
        return 0,0,0
    currLine = data[index]
    pastLine = data[index - dateDiff]
    #print(pastLine)
    pastPrice = float(pastLine[0])
    currPrice = float(currLine[0])
    percChange,diff = diffPerc(pastPrice, currPrice)
    pastVol = pastLine[2]
    
    return percChange, diff, pastVol


def getDayDiff(diff ,d, data, index):
    res = dateDifference(data, diff, index)
    DV = f'{diff}DayVol'
    DD = f'{diff}DayDiff'
    DPC = f'{diff}DayPercChange'

    if DV not in d:
        d[DV] = res[2]
    if DD not in d:
        d[DD] = res[1]
    if DPC not in d:
        d[DPC] = res[0]
    return d

def makeDataCats(dateDiffs):
    maxIndex = 0
    for key in dataCats:
        if dataCats[key] > maxIndex:
            maxIndex = dataCats[key]
    maxIndex += 1
    for diff in dateDiffs:
        DV = f'{diff}DayVol'
        DD = f'{diff}DayDiff'
        DPC = f'{diff}DayPercChange'
        adds = [DD, DPC, DV]
        for val in adds:
            dataCats[val] = maxIndex
            maxIndex += 1
    return dataCats


def addData(data, dateDiffs):
    index = 0 
    ### Dates differences
    for row in data:
        for dateDiff in dateDiffs:
            percChange, diff, pastVol = dateDifference(data, dateDiff, index)
            row.append(diff)
            row.append(percChange)
            row.append(pastVol)
        index += 1
    return data

def addLabels(data,forwardDate, changeAmount):
    count = len(data)
    dateDiff = forwardDate
    index = 0
    for row in data:
        forwardD = index + dateDiff
        currPrice = row[0]
        if (forwardD) < count:
            forwardLine = data[forwardD]
            forwardPrice = forwardLine[0]
            ### only doing bigger than for now
            forwardPrice = float(forwardPrice)
            currPrice = float(currPrice)
            if forwardPrice > currPrice + changeAmount:
                row.append(1)
            else:
                row.append(0)
        #else:
         #   row.append(0)
        index += 1
    return data


def getDataForStock(ticker, dateDiffs, forwardDate, changeAmount):
    res = list()
    for summary in rs.stocks.get_stock_historicals(ticker, interval = 'day', span = '5year'):
        #print(summary)
        line = getDataFromSum(summary)
        res.append(line)
    res = addData(res, dateDiffs)
    res = addLabels(res, forwardDate, changeAmount)
    #print(res)
    
    return res

def getData(tickerList, dateDiffs, forwardDate, changeAmount):
    res = list()
    for ticker in tickerList:
        temp = getDataForStock(ticker, dateDiffs, forwardDate, changeAmount)
        #print(temp)
        for row in temp:
            row.append(ticker)
            res.append(row)
    return res

def intoDF(data):
    dataCats['ticker'] = -1
    res = pd.DataFrame(data)
    cols = []
    for key in dataCats:
        cols.append(key)
    res.columns = cols
    return res


#print(getData(['AAPL', 'TSLA'])[:10])

# if __name__ == '__main__':
#     dateDiffs = [1,3,7]
#     data = getData(['AAPL', 'TSLA'], dateDiffs)
#     dataCats['ticker'] = -1
#     print(intoDF(data))


