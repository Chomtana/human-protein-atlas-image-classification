# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2 as cv
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
#import imgaug.augmenters as iaa
from random import randint

from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from tensorflow.keras import backend as K

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

print('human-protein-atlas-image-classification/train.csv')
print('human-protein-atlas-image-classification/sample_submission.csv')

print('Train n:', len(os.listdir('human-protein-atlas-image-classification/train')) ) # dir is your directory path
print('Test n:', len(os.listdir('human-protein-atlas-image-classification/test')) ) # dir is your directory path

train_df = pd.read_csv('human-protein-atlas-image-classification/train.csv')

IMG_SIZE = (224, 224)

'''
# Waiting for kaggle to upgrade imgaug to latest version
AUG_FN = iaa.Sequential([
    iaa.CropToFixedSize(width=IMG_SIZE[0], height=IMG_SIZE[1])
])
'''

def read_img(img_file, folder='train'):
    #print(img_file)
    img_red = cv.imread('/kaggle/input/human-protein-atlas-image-classification/'+folder+'/'+img_file+'_red.png', cv.IMREAD_GRAYSCALE) / 255
    img_green = cv.imread('/kaggle/input/human-protein-atlas-image-classification/'+folder+'/'+img_file+'_green.png', cv.IMREAD_GRAYSCALE) / 255
    img_blue = cv.imread('/kaggle/input/human-protein-atlas-image-classification/'+folder+'/'+img_file+'_blue.png', cv.IMREAD_GRAYSCALE) / 255
    # img_yellow = cv.imread('/kaggle/input/human-protein-atlas-image-classification/train/'+img_file+'_yellow.png', cv.IMREAD_GRAYSCALE) / 255
    img = np.rollaxis(np.array([img_red, img_green, img_blue]), 0, 3)
    
    # Crop image
    crop_y = randint(0, img.shape[0] - IMG_SIZE[1] - 1)
    crop_x = randint(0, img.shape[1] - IMG_SIZE[0] - 1)
    img = img[crop_y:crop_y+IMG_SIZE[1], crop_x:crop_x+IMG_SIZE[0]]
    
    return img

def generate_batch(n=64):
    while True:
        X = []
        Y = []

        for row in train_df.sample(n).itertuples():
            #print(row)
            img_file = row.Id
            img = read_img(img_file)
            label = row.Target.split()

            X.append(img)
            Y.append(label)

        mlb = MultiLabelBinarizer(classes=list(map(str, range(0,28))))
        Y = mlb.fit_transform(Y)
        #print(Y)

        X = np.array(X)
        Y = np.array(Y)

        #print(X.shape)

        yield (X, Y)
    
generate_batch()

original_model = tf.keras.applications.resnet_v2.ResNet50V2(weights=None, classes=28)

output_layer = Dense(28, activation='sigmoid', name='sigmoid_output')(original_model.layers[-2].output)

model = Model(inputs=original_model.input, outputs=output_layer)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def f1_macro(y_true, y_preds, thresh=0.5, eps=1e-20):
    preds_bin = y_preds > thresh # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true

    p = truepos.sum(axis=0) / (preds_bin.sum(axis=0) + eps) # sum along axis=0 (classes)
                                                            # and calculate precision array
    r = truepos.sum(axis=0) / (y_true.sum(axis=0) + eps)    # sum along axis=0 (classes) 
                                                            #  and calculate recall array

    f1 = 2*p*r / (p+r+eps) # we calculate f1 on arrays
    return np.mean(f1) # we take the average of the individual f1 scores at the very end!

def lr_schedule_fn(epoch):
    if epoch < 25:
        lr = 0.001
    elif epoch < 50:
        lr = 0.0005
    elif epoch < 75:
        lr = 0.0001
    elif epoch < 100:
        lr = 0.00005
    else:
        lr = 0.00001
    print('Learning rate: ', lr)
    return lr

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule_fn)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="ChomtanaProtein2.h5", save_best_only=True, mode='max', monitor='f1_macro', verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1_macro])

history = model.fit(generate_batch(), epochs=100, steps_per_epoch=64, callbacks=[lr_scheduler, checkpoint])