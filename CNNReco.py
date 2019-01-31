#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:23:20 2019

@author: sall
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

from keras import backend as K

K.set_image_dim_ordering('th')

import numpy as np
import os

from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#import json

import cv2

from matplotlib import pyplot as plt
#Import Keyboard for inputs
#import keyboard
#VLC Control
#import vlcControl
#Test
#import vlc
import vlcCtrl
#image dimensions
img_rows, img_cols = 200 , 200
#Channel Number
img_channels = 1


# Batch_size to train
batch_size = 32

## Number of output classes (change it accordingly)
## eg: In my case I wanted to predict 4 types of gestures (Ok, Peace, Punch, Stop)
## NOTE: If you change this then dont forget to change Labels accordingly
nb_classes = 5

# Number of epochs to train (change it accordingly)
nb_epoch = 15  #25

# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

#%%
#Path to the model or folder path
path = "./"
path1 = "./gestures"    #path of folder of images

path2 = './imgfolder_b'

WeightFileName = ["ori_4015imgs_weights.hdf5","bw_4015imgs_weights.hdf5","bw_2510imgs_weights.hdf5","./bw_weight.hdf5","./final_c_weights.hdf5","./semiVgg_1_weights.hdf5","/new_wt_dropout20.hdf5","./weights-CNN-gesture_skinmask.hdf5"]

# outputs
output = ["OK", "NOTHING","PEACE", "PUNCH", "STOP"]

jsonarray = {}
#%%
def update(plot):
    global jsonarray
    h = 450
    y = 30
    w = 45
    font = cv2.FONT_HERSHEY_SIMPLEX

    #plot = np.zeros((512,512,3), np.uint8)
    
    #array = {"OK": 65.79261422157288, "NOTHING": 0.7953541353344917, "PEACE": 5.33270463347435, "PUNCH": 0.038031660369597375, "STOP": 28.04129719734192}
    
    for items in jsonarray:
        mul = (jsonarray[items]) / 100
        #mul = random.randint(1,100) / 100
        cv2.line(plot,(0,y),(int(h * mul),y),(255,0,0),w)
        cv2.putText(plot,items,(0,y+5), font , 0.7,(0,255,0),2,1)
        y = y + w + 30

    return plot


#%%
# This function can be used for converting colored img to Grayscale img
# while copying images from path1 to path2
def convertToGrayImg(path1, path2):
    listing = os.listdir(path1)
    for file in listing:
        if file.startswith('.'):
            continue
        img = Image.open(path1 +'/' + file)
        #img = img.resize((img_rows,img_cols))
        grayimg = img.convert('L')
        grayimg.save(path2 + '/' +  file, "PNG")

#%%
def modlistdir(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #This check is to ignore any hidden files/folders
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist


# Load CNN model
def loadCNN(wf_index):
    global get_output
    model = Sequential()
    
    
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(img_channels, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    '''
    
    model.add(ZeroPadding2D((1,1),input_shape=(img_channels, img_rows, img_cols)))
    model.add(Conv2D(nb_filters , (nb_conv, nb_conv), activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Conv2D(nb_filters , (nb_conv, nb_conv), activation='relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.2))
    
    #model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(nb_filters , (nb_conv, nb_conv), activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    ##
    #model.add(Conv2D(nb_filters , (nb_conv, nb_conv), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=(2,2)))
    
    model.add(Dropout(0.3))
    model.add(Flatten())
    ###
    #model.add(Dense(128))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    '''
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    
    # Model summary
    model.summary()
    # Model conig details
    model.get_config()
    
    #from keras.utils import plot_model
    #plot_model(model, to_file='new_model.png', show_shapes = True)
    

    if wf_index >= 0:
        #Load pretrained weights
        fname = WeightFileName[int(wf_index)]
        print("loading ", fname)
        model.load_weights(fname)
    
    layer = model.layers[11]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    
    
    return model

# This function does the guessing work based on input images
def guessGesture(model, img):
    global output, get_output, jsonarray
    #Load image and flatten it
    image = np.array(img).flatten()
    
    # reshape it
    image = image.reshape(img_channels, img_rows,img_cols)
    
    # float32
    image = image.astype('float32') 
    
    # normalize it
    image = image / 255
    
    # reshape for NN
    rimage = image.reshape(1, img_channels, img_rows, img_cols)
    
    # Now feed it to the NN, to fetch the predictions
    #index = model.predict_classes(rimage)
    #prob_array = model.predict_proba(rimage)
    
    prob_array = get_output([rimage, 0])[0]
    
    #print prob_array
    
    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1
    
    # Get the output with maximum probability
    import operator
    
    guess = max(d.items(), key=operator.itemgetter(1))[0]
    prob  = d[guess]

    if prob > 60.0:
        print(guess + "  Probability: ", prob)
        if (guess == "NOTHING"):
            Instance = vlcCtrl.MediaPlayer()
            Instance.set_pause()
#        if (keyboard.is_pressed('space')):
#            print("Space was pressed")
        #Enable this to save the predictions in a json file,
        #Which can be read by plotter app to plot bar graph
        #dump to the JSON contents to the file
        
        #with open('gesturejson.txt', 'w') as outfile:
        #    json.dump(d, outfile)
        jsonarray = d
                
        return output.index(guess)

    else:
        return 1

#%%
def initializers():
    imlist = modlistdir(path2)
    
    image1 = np.array(Image.open(path2 +'/' + imlist[0])) # open one image to get size
    #plt.imshow(im1)
    
    m,n = image1.shape[0:2] # get the size of the images
    total_images = len(imlist) # get the 'total' number of images
    
    # create matrix to store all flattened images
    immatrix = np.array([np.array(Image.open(path2+ '/' + images).convert('L')).flatten()
                         for images in sorted(imlist)], dtype = 'f')
    

    
    print(immatrix.shape)
    
    input("Press any key")
    
    #########################################################
    ## Label the set of images per respective gesture type.
    ##
    label=np.ones((total_images,),dtype = int)
    
    samples_per_class = total_images / nb_classes
    print("samples_per_class - ",samples_per_class)
    s = 0
    r = samples_per_class
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class
    
    '''
    # eg: For 301 img samples/gesture for 4 gesture types
    label[0:301]=0
    label[301:602]=1
    label[602:903]=2
    label[903:]=3
    '''