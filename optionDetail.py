import csv
import robin_stocks.robinhood as rs
from datetime import datetime, date
import numpy as np
import pandas as pd


login = rs.login("Put email here",
                "Put Password Here", 
                expiresIn = 86400)


def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


def cleanS(s, d):
    res = []
    today = date.today()
    today = today.strftime("%Y-%m-%d")
    for o in s:
        if days_between(today, o['expiration_date']) < d:
            res.append(o)
        for key in o:
            print(key)
        break
    return res

def getEV(ticker, forwardDate, change):
    latest = rs.stocks.get_latest_price(ticker)
    options = rs.options.find_options_by_strike(ticker, strikePrice = int(float(latest[0])), optionType = 'call')

    s = sorted(options, key = lambda i: i['expiration_date'])
    cleaned = cleanS(s, forwardDate)
getEV('NKE', 100, 5)