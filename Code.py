#!/usr/bin/env python
# coding: utf-8

# <h1><center>Machine Learning Challenge: Image Classification</center></h1>

# Goutham Deekshit Indiran  | u195004 <br>
# Manav Mishra | u558101 <br>
# Sadjia Safdari | u265740

# **Load packages**

# In[ ]:


import numpy as np
get_ipython().run_line_magic('pylab', 'inline --no-import-all')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

## Neural nets
#.....................................
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.layers import Conv2D, Flatten, MaxPooling2D, GaussianNoise
from keras.callbacks import History
from keras.metrics import TopKCategoricalAccuracy, SparseTopKCategoricalAccuracy

## Image processing packages
import cv2
import os
from PIL import Image
import itertools
from numpy import asarray

## Matplot lib
#.....................................
import matplotlib.pyplot as plt

# Random Forest
import time
import random
from sklearn import ensemble 


# ## Task 1

# **Data**

# In[ ]:


with np.load("data/training-dataset.npz") as data:
        img = data["x"] # 97843200
        lbl = data["y"] # 124800


# In[ ]:


print(img.shape)

for im in range(10):
    image = img[im]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels)
    plt.show()


# ### 1.1 Neural Networks

# **Splitting data**

# In[ ]:


# Splitting data into train and validation +test with 80% for training data 

X_train, X_val_test, y_train, y_val_test = train_test_split(img, lbl, test_size=0.2,random_state=1) 

# Splitting validation + test 

X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.3, random_state=1)

# Visualising the shape of the train and test data
print(X_train.shape) 
print(X_test.shape) 
print(X_val.shape) 
print(y_test.shape) 
print(y_val.shape) 


# **Preprocessing data**

# In[ ]:


# Preprocessing the data so that it runs faster during fitting the model
# Preprocess the data (these are NumPy arrays)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32") / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32") / 255
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1).astype("float32") / 255
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")
y_val = y_val.astype("float32")


# **Transform data**

# In[ ]:


# One hot decoding 
onehot = LabelBinarizer() # transform categorical target to dummies to train the NN
Y_train = onehot.fit_transform(y_train)
Y_val   = onehot.fit_transform(y_val)
Y_test   = onehot.transform(y_test)


# **Fitting the model**

# In[ ]:


# Fitting the compiling and fitting the sequential model
#.....................................

es = EarlyStopping(monitor="val_loss")

model = Sequential() # initiates model
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(350, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(Y_train.shape[1], activation='softmax')) # We need to have as many units as classes, and softmax activation

# define parameters for training of the model    
optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy']) # for classification, the loss function should be categorical_crossentropy
history = model.fit(X_train, Y_train, epochs=100, batch_size=450, validation_data=(X_val, Y_val), verbose=1, callbacks = [es])


# **Evaluate**

# **Accuracy and Loss visualization**

# In[ ]:


# list all data in history

accuracy = history.history['accuracy']
#summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# **Predictions**

# In[ ]:


# Predicting the classes on test set and printing the accuracy
y_pred = model.predict_classes(X_test, verbose=1)
print(accuracy_score(y_test, y_pred)) # 0.889


# **Model Summary**

# In[ ]:


# Evaluating the model
model.evaluate(X_test,Y_test)
# Model summary
model.summary()
print((y_pred[:10]))
print((y_test[:10]).astype('int32'))


# ### 1.2 Random Forest

# In[ ]:


#Using Random Forest classifier
#oob_score This is a random forest cross validation method 
#This comes out very handy while scalling up a particular function from prototype to final dataset.
n_estimators = np.arange(100,1100,100)
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
classifier.fit(X_train,y_train)


# In[ ]:


# predict on test set
score = classifier.score(X_test,y_test) 
score 


# ## **Task 2**

# ### 2.1 Adding Gaussian noise to CNN model

# In[ ]:


# compiling and fitting the sequential model
#.....................................

es = EarlyStopping(monitor= 'val_loss')
optimizer = Adam(lr=0.0001)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding = 'same',strides=(1, 1),activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianNoise(0.5))
model.add(Flatten())
model.add(Dense(360, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(360, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(360, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(360, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(Y_train.shape[1], activation='softmax')) # We need to have as many units as classes, and softmax activation

# For classification, the loss function should be categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics = ['accuracy'])
history1 = model.fit(X_train, Y_train, epochs=100, batch_size = 450, validation_data = (X_val, Y_val), verbose=1, callbacks = [es])

# accuracy of 0.9253 > 23/100


# In[ ]:


# summarize history for accuracy
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
y_pred1 = model.predict_classes(X_test, verbose=1)
print(accuracy_score(y_test, y_pred1))


# **Predictions**

# In[ ]:


y_pred1 = model.predict_classes(X_test, verbose=1)
print(accuracy_score(y_test, y_pred1))
print(y_pred1)


# ### 2.2 Image processing

# In[ ]:


img_data = np.load('test-dataset.npy')
print(img_data.shape)


# **Saving images**

# In[ ]:


#make new folder
cwd = os.getcwd()
new_folder = "test_dataset_images"
folder = os.path.join(cwd, new_folder)
os.makedirs(folder)


# In[ ]:


for i in range(len(img_data)):
    data = Image.fromarray(img_data[i])
    if data.mode != 'RGB':
        data = data.convert('RGB')
    data.save(str(folder)+'\\test_image_'+str(i)+'.png')


# **Splitting image corrdinates**

# In[ ]:


def split_image(get_value):
    pass


# In[ ]:


def split_image(coord_list):
    new_coord=[]
    updated_coord=[]
    final_coord=[]
    for i in range(len(coord_list)):
            args = [iter(coord_list)] * 4
            new_coord = list(itertools.zip_longest(*args, fillvalue=None))
    return sorted(new_coord)


# **Predictions**

# In[ ]:


folder="test_dataset_images"
prediction_list = []
for filename in os.listdir(folder):
    coord_list=[]
    new_image_list=[]
    image_pred=[]
    sample_image = cv2.imread(os.path.join(folder,filename))
    if sample_image is not None:
        median_blr = cv2.medianBlur(sample_image, 3)
        plt.imshow(median_blr)
        median_blr.shape
        copy = median_blr.copy()
        gray = cv2.cvtColor(median_blr, cv2.COLOR_BGR2GRAY)
        prediction_image = gray
        thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        ROI_number = 0
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            ROI = img[y:y+h, x:x+w]
            if w > 9  or h > 10 :
                coord_list.append(x)
                coord_list.append(w)
                coord_list.append(y)
                coord_list.append(h)
            cv2.rectangle(copy,(x,y),(x+w,y+h),(36,255,12),1)
            ROI_number += 1
    imk=prediction_image
    coords= split_image(coord_list)
    plt.imshow(copy)
    plt.show()
    for i in range(len(coords)):
        if coords[i][1] >= 30 :
            new_w = coords[i][1] //2
            x_temp = 0
            for j in range(2):
                w = new_w
                imtest1=imk[coords[i][2]:coords[i][2]+coords[i][3],coords[i][0] + x_temp: coords[i][0] + x_temp + new_w]
                res1 = cv2.resize(imtest1, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
                new_image_list.append(np.asarray(res1))
                image_pred=np.array(new_image_list)
                image_pred = image_pred.reshape(image_pred.shape[0], 28, 28, 1).astype("float32") / 255
                preds1=model.predict_classes(image_pred, verbose=1)
                probs = model.predict_proba(image_pred)
                best_n = np.argsort(probs, axis=1)[:,-5:]
                best_transpose = np.transpose(best_n+1)
                best_n_str = np.char.zfill(best_transpose.astype(str), 2)
                join_best_n = ','.join([''.join(row) for row in best_n_str])
                print(i, len(coords))
                if i == len(coords)-1:
                    prediction_list.append([join_best_n])
                pred_label = preds1[:len(coords)+1]+1
                del coord_list[:]
                np.delete(image_pred,0,0)
                np.delete(best_n,0,0)
                x_temp = new_w
        else:
            imtest1=imk[coords[i][2]:coords[i][2]+coords[i][3],coords[i][0]:coords[i][0]+coords[i][1]]
            res1 = cv2.resize(imtest1, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            new_image_list.append(np.asarray(res1))
            image_pred=np.array(new_image_list)
            image_pred = image_pred.reshape(image_pred.shape[0], 28, 28, 1).astype("float32") / 255
            preds1=model.predict_classes(image_pred, verbose=1)
            probs = model.predict_proba(image_pred)
            best_n = np.argsort(probs, axis=1)[:,-5:]
            best_transpose = np.transpose(best_n+1)
            best_n_str = np.char.zfill(best_transpose.astype(str), 2)
            join_best_n = ','.join([''.join(row) for row in best_n_str])
            print(i, len(coords))
            if i == len(coords)-1:
                prediction_list.append([join_best_n])
            pred_label = preds1[:len(coords)+1]+1
            del coord_list[:]
            np.delete(image_pred,0,0)
            np.delete(best_n,0,0)


# **Save predictions into csv file**

# In[ ]:


import csv
with open('prediction.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for item in prediction_list:
        for in_item in item:
            it = in_item.strip().split(',')
            writer.writerow(it)

