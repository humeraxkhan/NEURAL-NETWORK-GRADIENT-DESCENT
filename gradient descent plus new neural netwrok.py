#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv(r"C:\Users\DELL\Downloads\insurance_data.csv")


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df[['age','affordibility']],df.bought_insurance,test_size=0.2,random_state=25)


# In[9]:


len(X_train)


# In[10]:


X_train_scaled =X_train.copy()
X_train_scaled['age']=X_train_scaled['age']/100

X_test_scaled =X_test.copy()
X_test_scaled['age']=X_test_scaled['age']/100


# In[11]:


X_test_scaled


# In[12]:


X_train_scaled # we converted age in points to bring age n affordibility on same scale


# In[13]:


model=keras.Sequential([
    keras.layers.Dense(1,input_shape=(2,),activation='sigmoid',kernel_initializer='ones', bias_initializer='zeros')
])


# In[14]:


model.compile(optimizer='adam'
             , loss='binary_crossentropy',
             metrics=['accuracy'])
model.fit(X_train_scaled,y_train,epochs=5000)


# In[15]:


model.evaluate(X_test_scaled,y_test)


# model.predict(X_test_scaled)

# In[16]:


model.predict(X_test_scaled)


# In[17]:


coef,intercept = model.get_weights()
coef,intercept


# In[18]:


import math
def sigmoid(x):
    return 1/(1+ math.exp(-x))


# In[19]:


def prediction_function(age,affordability):
    weighted_sum = coef[0]*age+coef[1]*affordability+intercept
    return sigmoid(weighted_sum)


# In[20]:


prediction_function(.47,1)


# In[21]:


#GRADIENT DESCENT DUNCTION IN PYTHON


# In[22]:


def log_loss(y_true,y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i,1-epsilon)for i in y_predicted]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))


# In[23]:


def sigmoid_numpy(X):
    return 1/(1+np.exp(-X))

sigmoid_numpy(np.array([12,0,1]))


# In[43]:


class myNN:
    def __init__(self):
        self.w1 = 1 
        self.w2 = 1
        self.bias = 0
        
    def fit(self, X, y, epochs, loss_thresold):
        self.w1, self.w2, self.bias = self.gradient_descent(X['age'],X['affordibility'],y, epochs, loss_thresold)
        print(f"Final weights and bias: w1: {self.w1}, w2: {self.w2}, bias: {self.bias}")
        
    def predict(self, X_test):
        weighted_sum = self.w1*X_test['age'] + self.w2*X_test['affordibility'] + self.bias
        return sigmoid_numpy(weighted_sum)

    def gradient_descent(self, age,affordability, y_true, epochs, loss_thresold):
        w1 = w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)
        for i in range(epochs):
            weighted_sum = w1 * age + w2 * affordability + bias
            y_predicted = sigmoid_numpy(weighted_sum)
            loss = log_loss(y_true, y_predicted)
            
            w1d = (1/n)*np.dot(np.transpose(age),(y_predicted-y_true)) 
            w2d = (1/n)*np.dot(np.transpose(affordability),(y_predicted-y_true)) 

            bias_d = np.mean(y_predicted-y_true)
            w1 = w1 - rate * w1d
            w2 = w2 - rate * w2d
            bias = bias - rate * bias_d
            
            if i%50==0:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
            
            if loss<=loss_thresold:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                break

        return w1, w2, bias
 




# In[25]:


gradient_descent(X_train_scaled['age'],X_train_scaled['affordibility'],y_train,1000, 0.4631)


# In[45]:


CustomModel = myNN()
CustomModel.fit(X_train_scaled, y_train , epochs = 8000 , loss_thresold = 0.4631)


# In[47]:


coef,intercept


# In[46]:


CustomModel.predict(X_test_scaled)


# In[48]:


model.predict(X_test_scaled)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




