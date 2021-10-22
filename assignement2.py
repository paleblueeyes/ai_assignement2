#!/usr/bin/env python
# coding: utf-8

# In[164]:


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


# In[165]:


#Read in the dataset
df = pd.read_csv("TSLA.csv")
#Converting date to a numeric value
df['Date'] = pd.to_datetime(df['Date'])
df['Date']=df['Date'].map(dt.datetime.toordinal)
#Droping all columns except the closing price
df = df.drop(columns=["Open", "Low","High", "Adj Close", "Volume"])


# In[166]:


#Creating a new feature that calculates the 50 day MA
df['MA50'] = df.rolling(window=50)["Close"].mean()


# In[167]:


#Plotting the MA50 against the closing price for illustration
ax1 = df.plot(kind='line', x='Date', y='Close', color='r')    
ax2 = df.plot(kind='line', x='Date', y='MA50', color='g', ax=ax1)


# In[168]:


#There is a high correlation between the closing price and the MA50, so we can try to train the
#model based on the MA50, 
corrMatrix.style.background_gradient(cmap='coolwarm')


# In[169]:


#SCALE THE VALUES
#x = df.values 
#scaler = preprocessing.MinMaxScaler().fit(x)
#x = scaler.transform(x)
#df = pd.DataFrame(x)
df = df.iloc[50: , :] # The first 50 rows is used to calculate MA50 and will be nan. Let's drop.
df.head()


# In[170]:


#df.tail()


# In[171]:


X = pd.DataFrame(df["Close"]) 
y = pd.DataFrame(df["MA50"])


# In[172]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[173]:


linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)  
Y_pred = linear_regressor.predict(X_train)  


# In[174]:


plt.scatter(X_train, y_train)             
plt.plot(X_train, Y_pred, color='red')    
plt.show()         
print( "MSE = "+str(metrics.mean_squared_error(y_train,Y_pred))) #Calculate MSE


# In[175]:


Y_pred = linear_regressor.predict(X_test)  
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red')
plt.show()
print( "MSE = "+str(metrics.mean_squared_error(y_test,Y_pred)))


# In[176]:


print(linear_regressor.predict([[800]]))


# In[177]:


#print(Y_pred)


# In[178]:


#print(df.iloc[1000])


# In[179]:


#print(Y_pred[1000][0])


# In[180]:


date_time_str = 'Jun 1 2017  1:33PM'
date_time_obj = dt.datetime.strptime(date_time_str,'%b %d %Y %I:%M%p')


# In[181]:


print(dt.datetime.toordinal(date_time_obj))


# In[198]:


#idx = df.index[df['Date']==736481].tolist()[0]
#print(idx)
#print(idx_minus_50)
close = df.loc[df['Date'] == 736481]['Close']
print(linear_regressor.predict([[close.values[0]]]))
print(close.values[0])
print(metrics.mean_squared_error([64], [68]))


# In[183]:


#df2 = df.iloc[idx_minus_50: idx, :]
#df2['MA502'] = df2.rolling(window=50)["Close"].mean()
#print(df2)


# In[ ]:




