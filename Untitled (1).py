#!/usr/bin/env python
# coding: utf-8

# In[132]:


import numpy
get_ipython().run_line_magic('pylab', 'inline --no-import-all')


# In[158]:


with np.load('training-dataset.npz')  as data:
    img = data['x']
    lbl = data['y']


# In[81]:


print(lbl)


# In[159]:


for im in range(10):
    image = img[im]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels)
    plt.show()


# In[ ]:





# In[160]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(img, lbl, test_size=0.3, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=1)


# In[161]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[162]:


print(y_test.shape)
print(X_val.shape)
print(y_train.shape)
print(X_val.shape)


# In[155]:


print(y_train)
print(X_train)


# In[163]:


X_train = X_train.reshape(43680, 784)
X_val = X_val.reshape(43680, 784)
X_test = X_test.reshape(37440, 784)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_val /= 255
X_test/= 255


# In[164]:


print(y_train)
print(X_val[1])


# In[165]:


from sklearn.preprocessing import LabelBinarizer
onehot = LabelBinarizer()
Y_train = onehot.fit_transform(y_train)
Y_val   = onehot.transform(y_val)
Y_test = onehot.transform(y_test)
print(Y_test.shape)


# In[166]:


print((Y_val))


# In[114]:


from keras import backend 

backend.clear_session()


# In[144]:



# Normalize the images.
X_train = (X_train / 255) - 0.5
X_val = (X_val / 255) - 0.5

# Flatten the images.
X_train = X_train.reshape((-1, 784))
X_val = X_val.reshape((-1, 784))

print(X_train.shape) 
print(y_train.shape) 


# In[167]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
baseline = LogisticRegression()
baseline.fit(X_train, y_train)
print(1-accuracy_score(y_val, baseline.predict(X_val)))


# In[170]:


print(accuracy_score(y_val, baseline.predict(X_val)))


# In[188]:


Ni = X_train.shape[1]
No = 26
Ns = y_train.size
alpha = 2
layer_size = int((Ns / (alpha * (Ni + No))))
print(layer_size)


# In[189]:


## Neural nets
#.....................................
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD

#.....................................
model = Sequential()
model.add(Dense(layer_size, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(layer_size, activation='relu'))
model.add(Dense(Y_train.shape[1], activation='softmax')) # We need to have as many units as classes, 
                                                             # and softmax activation
optimizer = Adam(lr=0.0001)
# For classification, the loss function should be categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1)


# In[190]:


from sklearn.metrics import accuracy_score
y_pred = model.predict_classes(X_val, verbose=1)
print(accuracy_score(y_val, y_pred))


# In[ ]:


model.fit(X_val, Y_val, epochs=100, batch_size=16, verbose=1)


# In[56]:


y_pred_train = np.argmax(model.predict(X_train), axis=-1) #Predict_classes raises an error and cannot be used soon, thus we use this instead
acc_train = round(accuracy_score(y_train, y_pred_train), 3)


# In[57]:


print(acc_train)


# In[47]:




