"""
Name: Mayank Bansal
Roll Number: B20156
Mobile No: +91963699345
Branch:CSE
"""

#import modules
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf as paf
import warnings
warnings.filterwarnings("ignore")

#read data
data=pd.read_csv('daily_covid_cases.csv')


#Q1a
#X axis
ticks=[0,180,360,540]
labels=['Feb-20','Aug-20','Feb-21','Aug-21']
#Plot data
plt.plot(data['Date'],data['new_cases'])
#give titles
plt.xlabel('Dates ->')
plt.ylabel('New confirmed cases')
plt.title('Line plot of New Cases')
#points marked on x-axis
plt.xticks(ticks,labels)
#show plot
plt.show()

#Q1b
data_1=data['new_cases'].shift(1)
#X axis
ticks=[0,180,360,540]
labels=['Feb-20','Aug-20','Feb-21','Aug-21']
#Plot data
plt.plot(data['Date'],data_1)
#give titles
plt.xlabel('Dates ->')
plt.ylabel('New confirmed cases')
plt.title('Line plot of New Cases with lag=1')
#points marked on x-axis
plt.xticks(ticks,labels)
#show plot
plt.show()
#Find autocorrelation
corr_1=sm.tsa.acf(data['new_cases'],nlags=1)
#print result
print("Autocorrelation of data with lag=1:",corr_1[1])

#Q1c
#scatter data and plot data
plt.scatter(data['new_cases'],data_1)
#give title
plt.xlabel("Actual time sequence")
plt.ylabel("Time sequence with lag=1")
plt.title("Satter plot of time sequence with lag=1")
#show plot
plt.show()

#Q1d
#corr list
p_corr=[]
for i in range(1,7):
    #corr function
    autocorrelation=sm.tsa.acf(data['new_cases'],nlags=i)
    #add data in list
    p_corr.append(autocorrelation[i])
#print list
print("AR list:",p_corr)    
#lag list
day=[1,2,3,4,5,6]
#plot data
plt.plot(day,p_corr)
#give label and title
plt.title("Autocorrelation Graph")
plt.xlabel("Lags->")
plt.ylabel("Autocorrelation")
#show
plt.show()

#Q1e
#use inbuilt func
paf(data['new_cases'],lags=25)
#show plot
plt.show()
print()

#Q2a
#read data
data2 = pd.read_csv('daily_covid_cases.csv',index_col=['Date'],sep=',')
#define test size
test_size = 0.35
#test size
X = data2.values
tst_sz = math.ceil(len(X)*test_size)
#divide data
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]
#build model on train data with lag=5
window = 5
model = AutoReg(train, lags=window)
model_fit = model.fit()
#find coefficient
coef = model_fit.params
#print result
print("Coefficient:",coef.tolist())
#Q2b
#define history for future data
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
#prediction list
predictions = []
#predict future data
for t in range(len(test)):
	length = len(history)
    #define history for future test result
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
    #predict data
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	#append data and predictions
	predictions.append(yhat)
	history.append(test[t])
#scatter data
plt.scatter(test,predictions)
#titles and labels
plt.xlabel("Actual data")
plt.ylabel("Predicted data")
plt.title("Scattered data")
#show plot
plt.show()

#plot test and predicted data
plt.plot(test,label ='Actual data')
plt.plot(predictions, color='red',label ='Predicted data')
#X axis
ticks=[1,50,100,150,200]
labels=['Mar-21','Apr-21','Jun-21','Jul-21','Sep-21']
#points marked on x-axis
#points marked on x-axis
plt.xticks(ticks,labels)
#show legend
plt.legend()
#show plot
plt.show()
print(test[0])

#RMSE func
rmse = math.sqrt(mean_squared_error(test, predictions))
#show RMSE %
print("Test RMSE%  --->")
print('%.3f' % (rmse*len(test)*100/sum(test)))

#MAPE func
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#print MAPE of test data
print("Test MAPE --->")
print(mean_absolute_percentage_error(test,predictions))
print()

#Q3
#read data
data3 = pd.read_csv('daily_covid_cases.csv',index_col=['Date'],sep=',')
#define test size
test_size = 0.35
#test size
X = data3.values
tst_sz = math.ceil(len(X)*test_size)
#divide data
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]
#build model on train data with lag=5
window = 5
model = AutoReg(train, lags=5)
model_fit = model.fit()
laglist=[1,5,10,15,25]
RMSE=[]
Mape=[]
#find coefficient
coef = model_fit.params
for window in (1,5,10,15,25):
    print("Lag=",window)
    model = AutoReg(train, lags=window)
    model_fit = model.fit()
    #find coefficient
    coef = model_fit.params
    #Q2b
    #define history for future data
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    #prediction list
    predictions = []
    #predict future data
    for t in range(len(test)):
    	length = len(history)
        #define history for future test result
    	lag = [history[i] for i in range(length-window,length)]
    	yhat = coef[0]
        #predict data
    	for d in range(window):
    		yhat += coef[d+1] * lag[window-d-1]
    	#append data and predictions
    	predictions.append(yhat)
    	history.append(test[t])
    #RMSE func
    rmse = math.sqrt(mean_squared_error(test, predictions))
    #show RMSE %
    print("Test RMSE%  --->")
    print('%.3f' % (rmse*len(test)*100/sum(test)))
    #append RMSE
    RMSE.append('%.3f' % (rmse*len(test)*100/sum(test)))
    #MAPE func
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    #print MAPE of test data
    print("Test MAPE --->")
    print(mean_absolute_percentage_error(test,predictions))
    #append MAPE
    Mape.append(mean_absolute_percentage_error(test,predictions))
    print()
#plot data
plt.bar(laglist,RMSE)
plt.title("RMSE%")
plt.show()
plt.bar(laglist,Mape)
plt.title("MAPE")
plt.show()

#Q4
data4 = pd.read_csv('daily_covid_cases.csv',index_col=['Date'],sep=',')
test_size = 0.35
#test size
X = data4.values
tst_sz = math.ceil(len(X)*test_size)
#divide data
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]
lag_value=0
alpha=pd.DataFrame(train)
for lags in range(0,397):
    data_lag = pd.Series(alpha.iloc[0:397-lags][0]) ##lag-time series 
    data_t = pd.Series(alpha.iloc[lags:397][0])  ##given time squence 
    data_t.index=[i for i in range(397-lags)]
    ar=data_t.corr(data_lag)
    if(np.absolute(ar)<2/np.sqrt(397-lags)):
        break
    else:
        lag_value=lags
print("Lag Value=",lag_value)
#build model on train data with lag=5
window = lag_value
model = AutoReg(train, lags=window)
model_fit = model.fit()
#find coefficient
coef = model_fit.params
#define history for future data
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
#prediction list
predictions = []
#predict future data
for t in range(len(test)):
	length = len(history)
    #define history for future test result
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
    #predict data
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	#append data and predictions
	predictions.append(yhat)
	history.append(test[t])
#RMSE func
rmse = math.sqrt(mean_squared_error(test, predictions))
#show RMSE %
print("Test RMSE%  --->")
print('%.3f' % (rmse*len(test)*100/sum(test)))
#MAPE func
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#print MAPE of test data
print("Test MAPE --->")
print(mean_absolute_percentage_error(test,predictions))
