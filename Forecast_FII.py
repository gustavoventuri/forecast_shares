# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 13:05:01 2020
running with Spider(Python 3.7)

@author: gustavo venturi

DISCLAIMER:
    THIS IS NOT A RECOMENDATION TO BUY OU SELL SHARES.
    YOU MUST DO YOUR OWN VERIFICATION BEFORE BUY OR SELL, IT'S UP TO YOU!
    

description: forecast with sklearn to shares from IBOVESPA.

Before start, you need input at least 5 tickers to compare shares.

The ticker input, is using YAHOO FINANCE names.

Check the variable label_ticker, if you input a 5 or 6 character lenght, if 5 change to five.


##I'M NOT IDENT THIS CODE. TAKE CARE WITH GARBAGE....
"""

import pandas as pd
import datetime
import pandas_datareader.data as web

from pandas import Series, DataFrame

import math

import numpy as np

from sklearn import preprocessing

from datetime import date

#####
####
today = date.today()

start = datetime.datetime(2015, 1, 1)
end = today



################
################
ticker = "CARE11"+".SA"
###############
###############


label_ticker = ticker[:6]
 

df = web.DataReader(ticker, 'yahoo', start, end)
df.tail()

close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

###
import matplotlib.pyplot as plt
from matplotlib import style

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(10, 5))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label=ticker)
mavg.plot(label='mavg')
plt.legend()

rets = close_px / close_px.shift(1) - 1
rets.plot(label='return')

########################
#######first_run########
########################

dfcomp = web.DataReader([ticker,'BRCR11.SA',
'CARE11.SA',
'CXRI11.SA',
'FIIB11.SA',
'FLMA11.SA',
'HGRU11.SA',
'JSRE11.SA',
'KNRI11.SA',
'MFII11.SA',
'MXRF11.SA',
'PABY11.SA',
'RBRD11.SA',
'RECT11.SA'
],'yahoo',start=start,end=end)['Adj Close']

dfcomp.tail()

dfcomp.rename(columns={ ticker: label_ticker,
                    'BRCR11.SA':'BRCR11',
                    'CARE11.SA':'CARE11',
                    'CXRI11.SA':'CXRI11',
                    'FIIB11.SA':'FIIB11',
                    'FLMA11.SA':'FLMA11',
                    'HGRU11.SA':'HGRU11',
                    'JSRE11.SA':'JSRE11',
                    'KNRI11.SA':'KNRI11',
                    'MFII11.SA':'MFII11',
                    'MXRF11.SA':'MXRF11',
                    'PABY11.SA':'PABY11',
                    'RBRD11.SA':'RBRD11',
                    'RECT11.SA':'RECT11'
                      }, inplace=True)
dfcomp.tail()

#Correlation  between tickers

retscomp = dfcomp.pct_change()
corr = retscomp.corr()
corr

#Choose 2 tickers to correlate 
###OPTIONAL RUN only if you want to compare. 
#### TO RUN> CHANGE THE TICKERS BEFORE
plt.scatter(retscomp.CARE11, retscomp.FLMA11)
plt.xlabel(retscomp.CARE11)
plt.ylabel(retscomp.FLMA11)


###########
###########
#Making a correlation matrix - all tickers
pd.plotting.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10))

#Ploting a heatmap
plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)

#eg. BLACK IS BAD CORRELATION - WHITE IS GOOD

############
############
#Risk matrix

plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
 
#eg. Higher values corresponds higher risk.
 
    

#####################
#####################
#####################    

#Prediction price...

    
dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

dfreg

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

#######################
####################### RUN ONCE 
#######################

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


#######################
#######################
#######################


# Linear Regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X, y)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X, y)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X, y)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X, y)

#Evaluation
confidencereg = clfreg.score(X, y)
confidencepoly2 = clfpoly2.score(X,y)
confidencepoly3 = clfpoly3.score(X,y)
confidenceknn = clfknn.score(X, y)

print("LR:  " + str(confidencereg))
print("QR2: " + str(confidencepoly2))
print("QR3: " + str(confidencepoly3))
print("KNN: " + str(confidenceknn))

#######################################
#######################################
#######################################

from sklearn import svm

forecast_set = clfpoly3.predict(X_lately) ###Check the best regression and change the classification
dfreg['Forecast'] = np.nan


#######GRAND FINALE!

#plot results
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(ticker)
plt.show()

dfreg.tail(20)