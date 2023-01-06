#!/usr/bin/env python
# coding: utf-8

# Detecting Parkinson Disease

# Work Flow:
# 1. Parkinson's data
# 2. Data pre-processing
# 3. Split data into Train and Test data
# 4. Use ML: Support Vector Machine classifier
# 5. After training the model with classifier we will get the updated data
# 6. Predict whether the person has disease or not
# 

# In[1]:


#Import dependency/Libraries
import numpy as np
import pandas as pd #for created dataframes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm 
from sklearn.metrics import accuracy_score


# Data collection and preprocessing

# In[2]:


#load data from csv file to pandas dataframe
data= pd.read_csv(r'C:\Users\APURVA\Downloads\parkinsons.csv')


# In[3]:


# print the head(frist 5 rows)
data.head(10)


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


#checking for missing values in each column
data.isnull().sum()


# In[7]:


#getting statistical value
data.describe()


# In[8]:


#distribution of target variable
data['status'].value_counts()


# 1 is parkinson person
# 0 is no parkinson

# In[9]:


#grouping the data based on target variable
data.groupby('status').mean()


# Data preprocessing

# In[10]:


#Separate the features and target
X = data.drop(columns = ['name', 'status'], axis = 1)
Y = data['status']


# Spliting the data to training data and target

# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state = 2)


# In[12]:


print(X.shape, X_train.shape, X_test.shape)


# Data Standardization

# In[14]:


scaler = StandardScaler()


# In[15]:


scaler.fit(X_train)


# In[16]:


X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


# In[17]:


print(X_train)


# In[19]:


#Model Training --> Support Vector Machine

model = svm.SVC(kernel='linear')


# In[20]:


#training the SVM model with training data 
model.fit(X_train,Y_train)


# In[22]:


#Model Evaluation
#Accuracy score on training data 
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train,X_train_prediction)


# In[23]:


print(training_data_accuracy)


# In[24]:


#Model Evaluation
#Accuracy score on test data 
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test,X_test_prediction)


# In[25]:


print(test_data_accuracy)


# In[27]:


#Build a predictive system
input_data = (104.4,206.002,77.968,0.00633,0.00006,0.00316,0.00375,0.00948,0.03767,0.381,0.01732,0.02245,0.0378,0.05197,0.02887,22.066,0.522746,0.737948,-5.571843,0.236853,2.846369,0.219514)

input_data_as_numpy_array = np.asarray(input_data)

#reshape
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

#standardize data 
std_data = scaler.transform(input_data_reshape)

prediction = model.predict(std_data)
print(prediction)


# In[ ]:


input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data(tuple) to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")

