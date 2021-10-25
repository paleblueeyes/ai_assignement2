#!/usr/bin/env python
# coding: utf-8

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[14]:


#Read in the dataset
df = pd.read_csv("TSLA.csv")
#Converting date to a numeric value
df['Date'] = pd.to_datetime(df['Date'])
df['Date']=df['Date'].map(dt.datetime.toordinal)
#Droping all columns except the closing price
df = df.drop(columns=["Open", "Low","High", "Adj Close", "Volume"])


# In[15]:


#Creating a new feature that calculates the 50 day MA
df['MA50'] = df.rolling(window=50)["Close"].mean()


# In[16]:


#Plotting the MA50 against the closing price for illustration
ax1 = df.plot(kind='line', x='Date', y='Close', color='r')    
ax2 = df.plot(kind='line', x='Date', y='MA50', color='g', ax=ax1)


# In[100]:


df = df.iloc[50: , :] # The first 50 rows is used to calculate MA50 and will be nan. Let's drop.


# In[20]:


X = pd.DataFrame(df["Close"]) 
y = pd.DataFrame(df["MA50"])


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[22]:


linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)  
Y_pred = linear_regressor.predict(X_train)  


# In[23]:


plt.scatter(X_train, y_train)             
plt.plot(X_train, Y_pred, color='red')    
plt.show()         
print( "MSE = "+str(metrics.mean_squared_error(y_train,Y_pred))) #Calculate MSE


# In[24]:


Y_pred = linear_regressor.predict(X_test)  
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red')
plt.show()
print( "MSE = "+str(metrics.mean_squared_error(y_test,Y_pred)))


# In[86]:


# Write date on this form Jun 1 2017  1:33PM
def predictOnDate(date):
    date_time_obj = dt.datetime.strptime(date,'%b %d %Y')
    date_time_obj = date_time_obj.toordinal()
    close = df.loc[df['Date'] == date_time_obj]['Close']
    prediction = linear_regressor.predict([[close.values[0]]])[0][0] # i know this syntax is so off prob some better way to do it
    close = close.values[0]
    print("CLOSING PRICE: " +  str(close) + "\n" + "PREDICTED PRICE: " + str(prediction))
    print(metrics.mean_squared_error([close], [prediction]))


# In[101]:


predictOnDate('Dec 31 2019')


# In[ ]:


#For some reason it only works until Dec 31 2019, not quite sure why
#In real life MA50 obviously wouldn't be the best way to train the algo
# but i assume that is out of the scope of the task

