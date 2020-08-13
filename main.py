"""This is an Ensamble based Neural Network model
for implementation of Anomaly detection and classification
on the Humerus data samples of the MURA dataset by Stamford
Machine Learning Group."""

"""Authors:
Md Sahil Hassan (Dept. of Electrical & Computer Engineering)
Mithun Ghosh (Dept. of Systems & Industrial Engineering)
Universiy of Arizona"""


from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os
import tensorflow as tf

"""User defined Functions"""
from make_nn import make_nn
from errcalc import errcalc
from train_nn import train_nn
from predict_nn import predict_nn
from kappa_calc import kappa_calc
from weighted_error import weighted_error
from ens_pred_nn import ens_pred_nn
from sampling import sampling
from quantizer import quantizer


"""Load Train Data"""
#code for reading data
imgdir = []   #shall contain the full path of the image
ytr = []         #shall contain label of each image
#xtr = np.matrix([[]])         #shall contain each image

#xtr = np.zeros((257,262144))   #modified
xtr = np.zeros((1300,10000))

datatype = ['train', 'valid']
studylabel = {'negative':0, 'positive':1}

BASE_DIR = 'mura_humerus'

traindir = BASE_DIR + '/' + datatype[0]  #shows training directory

patients = os.listdir(traindir)  #all the patient ID's are now into this

i = 0
for patient in patients:
    directory = traindir + '/' + patient    #patient directory train/patientxxxx/
    studies = os.listdir(directory)         #creates list of studies in each patient directory
    for study in studies:
        images = os.listdir(directory + '/' + study)
        for image in images:
            imgdir = np.append(imgdir, directory + '/' + study + '/' + image)    #add image addresses in array
            ytr = np.append(ytr, studylabel[study.split('_')[1]])        #keep adding labels of each images
            file = mpimg.imread(imgdir[i])
            if (len(np.shape(file))>2):
                pixels = np.matrix(file[:,:,0])     #takes matrix format of one image
                pixels = np.reshape(pixels, (1,len(pixels)*len(pixels.T)))   #flattens the image to 1-D
            else:
                pixels = np.matrix(file)
                pixels = np.reshape(pixels, (1,len(pixels)*len(pixels.T)))
                
            if (np.shape(pixels)[1] < 10000):    #modified from 262144
                zeroarray = np.matrix(np.zeros((1,(10000-np.shape(pixels)[1]))))
                pixels = np.concatenate((pixels,zeroarray), axis=1)
                #print(pixels)
                #xtr = np.append(xtr, np.array(file[:,:,0]), axis=0)
                ###xtr = np.concatenate(xtr,pixels, axis=1)               #load and add images in 
                xtr[i] = pixels               #load and add images in 
                #print ('i from if=', i)
            else:
                xtr[i] = np.matrix(pixels[:,0:10000])     #modified from 262144
                #print(pixels)
                #print ('i from else=', i)
                
            i=i+1
            
xtrain = xtr[0:i,:]
ytrain = ytr


"""Normalize data"""
for i in range (len(xtrain)):
    if np.amax(xtrain[i]) != 0:
        xtrain[i] = xtrain[i] / np.amax(xtrain[i])

"""Eliminating all zero data"""
xtr = np.zeros((np.shape(xtrain)))
ytr = []
j = 0
i = 0
while i < (len(ytrain)):
    if np.amax(xtrain[i]) != 0:
        xtr[j] = xtrain[i]
        ytr.append(ytrain[i])
        j = j+1
    i = i+1
    
xtr = xtr[0:j,:]


"""Main program body"""
"""Sampling based ensamble"""

###Step 1: Load Training Data
#Done, not in a function.



### Adaboost flow
##=================================

# x,y = traindata, trainlabel
x,y = xtr, ytr

# initial = D = 1/n
nbsample = len(ytr)
D = (1/nbsample)* np.ones((nbsample,))

# variable to hold neural nets
nn_cluster = []

#variable to hold alpha values
alpha = []

for t in range (5):
    #1. create classifier
    shallow_nn = make_nn(10000, 8000, 7000, 6000, 5000, 4000, 2500, 1250, 650, 300, 100, 1)

    #2. Train using xtr
    train_nn(shallow_nn,x,y, 30, 50, True)

    #3. Predict on xtr
    ypred, __ = predict_nn(shallow_nn, x, y)

    #4. calculate epsilon
    epsilon = weighted_error(y,ypred,D)

    #5. Calculate alpha
    alph = 0.5*math.log10((1-epsilon)/epsilon)

    #6. update D
    D = distr_upd(D, epsilon, alph, y, ypred)

    #7. append the already made classifier
    nn_cluster.append(shallow_nn)

    #8. append the alpha values captured
    alpha.append(alph)

    #9. Sample data for next iteration
    sampled_ind = sampling(D,nbsample)
    x , y = x[sampled_ind,:], np.array(y)[sampled_ind]


### By this point, the classifier should be prepared, now we apply the predict function

ytr_predict = ens_pred_nn(nn_cluster, xtr, ytr, alpha)

### Calculating Error
train_error = errcalc(ytr, ytr_predict)

### Calculating Kappa

train_kappa = kappa_calc(ytr, ytr_predict)


## Printing Results

print('**************==============******************')
print('Training error is ', train_error, ' and kappa is ', train_kappa)


"""Load Validation Data"""
#code for reading data
imgdir = []   #shall contain the full path of the image
yts = []         #shall contain label of each image
#xtr = np.matrix([[]])         #shall contain each image

#xtr = np.zeros((257,262144))   #modified
xts = np.zeros((1300,10000))

#datatype = ['train', 'valid']
#studylabel = {'negative':0, 'positive':1}

BASE_DIR = 'mura_humerus'

validdir = BASE_DIR + '/' + datatype[1]  #shows training directory

patients = os.listdir(validdir)  #all the patient ID's are now into this

i = 0
for patient in patients:
    directory = validdir + '/' + patient    #patient directory train/patientxxxx/
    studies = os.listdir(directory)         #creates list of studies in each patient directory
    for study in studies:
        images = os.listdir(directory + '/' + study)
        for image in images:
            imgdir = np.append(imgdir, directory + '/' + study + '/' + image)    #add image addresses in array
            yts = np.append(yts, studylabel[study.split('_')[1]])        #keep adding labels of each images
            file = mpimg.imread(imgdir[i])
            if (len(np.shape(file))>2):
                pixels = np.matrix(file[:,:,0])     #takes matrix format of one image
                pixels = np.reshape(pixels, (1,len(pixels)*len(pixels.T)))   #flattens the image to 1-D
            else:
                pixels = np.matrix(file)
                pixels = np.reshape(pixels, (1,len(pixels)*len(pixels.T)))
                
            if (np.shape(pixels)[1] < 10000):    #modified from 262144
                zeroarray = np.matrix(np.zeros((1,(10000-np.shape(pixels)[1]))))
                pixels = np.concatenate((pixels,zeroarray), axis=1)
                #print(pixels)
                #xtr = np.append(xtr, np.array(file[:,:,0]), axis=0)
                ###xtr = np.concatenate(xtr,pixels, axis=1)               #load and add images in 
                xts[i] = pixels               #load and add images in 
                #print ('i from if=', i)
            else:
                xts[i] = np.matrix(pixels[:,0:10000])     #modified from 262144
                #print(pixels)
                #print ('i from else=', i)
                
            i=i+1

#print(i)
xtest = xts[0:i,:]
#print(xtest.shape)
ytest = yts[0:i]
#print(i)
#print(np.shape(ytest))


"""Normalize data"""
for i in range (len(xtest)):
    if np.amax(xtest[i]) != 0:
        xtest[i] = xtest[i] / np.amax(xtest[i])
        
        
"""Eliminating all zero data"""
xts = np.zeros((np.shape(xtest)))
yts = []
j = 0
i = 0
while i < (len(ytest)):
    if np.amax(xtest[i]) != 0:
        xts[j] = xtest[i]
        yts.append(ytest[i])
        j = j+1
    i = i+1
    
xts = xts[0:j,:]


"""Evaluating test performance"""

### By this point, the classifier should be prepared, now we apply the predict function

yts_predict = ens_pred_nn(nn_cluster, xts, yts, alpha)

### Calculating Error
test_error = errcalc(yts, yts_predict)

### Calculating Kappa

test_kappa = kappa_calc(yts, yts_predict)


## Printing Results

print('**************==============******************')
print('Validation error is ', test_error, ' and kappa is ', test_kappa)

