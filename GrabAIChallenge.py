
# coding: utf-8

# # GRAB AI for S.E.A. Challenge
# Installing important libraries
# In[1]:


## important import libraries, if not installed, do sudo pip install <library_name> in terminal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os

# The code requires the safety folder dataset to be in the same directory which contains the notebook 
# In[3]:


os.getcwd()


# In[4]:


os.chdir(os.getcwd()+'/safety/labels')


# In[5]:


labels_df = pd.read_csv("part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv")


# In[6]:


os.chdir(os.getcwd()+'/../features/')

# Reading the dataset...
# In[7]:


df1 = pd.read_csv("part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df2 = pd.read_csv("part-00001-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df3 = pd.read_csv("part-00002-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df4 = pd.read_csv("part-00003-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df5 = pd.read_csv("part-00004-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df6 = pd.read_csv("part-00005-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df7 = pd.read_csv("part-00006-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df8 = pd.read_csv("part-00007-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df9 = pd.read_csv("part-00008-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df10 = pd.read_csv("part-00009-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")


# In[8]:


features_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10])


# In[9]:


features_df.dropna(inplace=True)

# Created a absolute acceleration variable which is basically the vector sum of acceleration in x and y direction. 
# In[10]:


features_df['absolute_acc'] = np.sqrt(features_df['acceleration_x']**2+features_df['acceleration_z']**2)

# There were minimum 120 time slices of data point for each booking number. So, I sort the data points first according to their booking ID (i.e, same booking ID are grouped together) then time slices with same booking ID are sorted in descending order of their absolute acceleration (sqrt((acc_x)**2+(acc_y)**2)). Top 120 data points are selected for training as higher absolute acceleration datapoints have higher chance to decide the safety of the trip.
# In[11]:


features_df.sort_values(['bookingID', 'absolute_acc'], ascending=[True, False],inplace=True)


# In[12]:


features_df.reset_index(inplace=True)


# In[13]:


features_df.drop('index',axis=1,inplace=True)

# Preprocessing the data to normalise all the variables.
# In[14]:


from sklearn import preprocessing
    
features_df[['acceleration_x','acceleration_y','acceleration_z','gyro_x','gyro_y','gyro_z','Speed','absolute_acc']] = preprocessing.scale(features_df[['acceleration_x','acceleration_y','acceleration_z','gyro_x','gyro_y','gyro_z','Speed','absolute_acc']])


# In[15]:



a = np.zeros(((np.size(labels_df['label'])),120,8))

# Keeping acceleration in all axes, gyro readings in all axes, speed and absolute acceleration as 8 variables
# In[16]:


c=0
for i in labels_df['bookingID']:
    a[c][:][:]=features_df[features_df['bookingID']==i][['acceleration_x','acceleration_y','acceleration_z','gyro_x','gyro_y','gyro_z','Speed','absolute_acc']][0:120]
    c=c+1


# In[17]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence


# In[18]:


from sklearn.model_selection import train_test_split

# Doing a train test split of 75 - 25%
# In[19]:


X_train, X_test, y_train, y_test = train_test_split(a, labels_df['label'], test_size=0.25, random_state=42)


# # Sequence classification using Convolutional and Recurrent Neural Network
# The problem can be treated as classification of time sequence of telemantics data points. 
# In[20]:


model = Sequential()
model.add(Conv1D(input_shape=(120,8),filters=30, kernel_size=3,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=25, batch_size=100)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[21]:


y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
y_pred[y_pred <= 0.5] = 0
y_pred[y_pred > 0.5] = 1
accuracy_score(y_test,y_pred)

