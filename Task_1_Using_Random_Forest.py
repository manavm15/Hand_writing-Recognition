#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pylab as pl


# In[ ]:


with np.load('training-dataset.npz') as data:
    img = data['x']
    lbl = data['y']
print(img.shape)


# In[ ]:


for im in range(10):
    image = img[im]
    label = lbl[im]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.subplot(2,5, im+1)
    plt.axis("off")
    plt.imshow(pixels)
    plt.title('%i' %label)
    plt.show()


# In[ ]:


import random
from sklearn import ensemble #psychic learn we use ensemble - it has random forest classifier


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(img, lbl, test_size=0.2, random_state=1) #first split train 80, val+test 20

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.3
                   , random_state=1) 


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(X_val.shape)


# In[ ]:


#Using Random Forest classifier
#oob_score This is a random forest cross validation method 
#This comes out very handy while scalling up a particular function from prototype to final dataset.
n_estimators = np.arange(100,1100,100)
import time
t0 = time.time()
for i in n_estimators: 
    classifier = ensemble.RandomForestClassifier(n_estimators = i, oob_score = True, n_jobs = -1,random_state =50)
    classifier.fit(X_train,y_train)
    # evaluating the model on the validation set and checking best value of k
    score = classifier.score(X_val,y_val)
    print("No. of trees = %d, accuracy=%.2f%%" % (i, score * 100))
t1 = time.time()
total = t1-t0
print(total) 


# In[ ]:


#training model on train data with best value of n_estimators
classifier = ensemble.RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =50)
import time
t0 = time.time()
classifier.fit(X_train,y_train)
t1 = time.time()
total = t1-t0
print(total) 
#takes approx. 20 minutes


# In[ ]:


# predict on test set
score = classifier.score(X_test,y_test)
score


# In[ ]:


im=7 #try changing different values 
view_img = img[im]
label = lbl[im]
view_img = np.array(view_img, dtype='float')
pixels = view_img.reshape((28, 28))
plt.axis("off")
plt.imshow(pixels)
plt.title('%i' %label)
plt.show()


# In[ ]:


#check the prediction for the above image
classifier.predict(img[[7]])


# In[ ]:


#Task 2 


# In[ ]:


#Adding noise to image


# In[ ]:


original_image = img
for im in range(2):
    img_test = original_image[im]
    img_test = np.array(img_test, dtype='float64')
    pixels = img_test.reshape((28,28))
    plt.imshow(pixels)
    plt.show()


# In[ ]:


from skimage.util import random_noise


# In[ ]:


test_image = np.array(original_image, dtype='float64')
image_with_noise = random_noise(test_image, mode="s&p") #salt and pepper noise


# In[ ]:


for im in range(2):
    img_test = image_with_noise[im]
    img_test = np.array(img_test, dtype='float64')
    pixels = img_test.reshape((28,28))
    plt.imshow(pixels)
    plt.show()


# In[ ]:


X_train_noise, X_val_noise, y_train_noise, y_val_noise = train_test_split(image_with_noise, lbl, test_size=0.2, random_state=1) #first split train 80, val+test 20

X_val_noise, X_test_noise, y_val_noise, y_test_noise = train_test_split(X_val_noise, y_val_noise, test_size=0.3
                   , random_state=1) 


# In[ ]:


classifier = ensemble.RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =50)
import time
t0 = time.time()
classifier.fit(X_train_noise,y_train_noise)
score = classifier.score(X_val_noise,y_val_noise)

t1 = time.time()
total = t1-t0
print(total) 


# In[ ]:


score


# In[ ]:


# predict on test set
score = classifier.score(X_test_noise,y_test_noise)
score

