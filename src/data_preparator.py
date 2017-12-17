'''
Created on Dec 13, 2017

Prepares the dataset as numpy arrays for training

Assumes fer2013 dataset is in ../data/fer2013

@author: Vladimir Petkov
'''

import pandas as pd
import numpy as np
from keras.utils import to_categorical


def flat_to_matrix(pixels_string, size=(48, 48)):
    pixels_arr = np.array(map(int, pixels_string.split()))
    return pixels_arr.reshape(size)


def _get_data(data_type=None, rows_to_read=None):
    df = pd.read_csv('../data/fer2013.csv', sep=',', nrows=rows_to_read)

    if data_type is not None:
        df = df[df.Usage == data_type]
    df['pixels'] = df.pixels.apply(lambda x: flat_to_matrix(x))

    x = np.array([matrix for matrix in df.pixels])
    y = np.array([emotion for emotion in df.emotion])
    
    x = x.reshape(-1, x.shape[1], x.shape[2], 1)
    y = to_categorical(y)
    
    return x, y


def get_test_data():
    return _get_data('PublicTest')


def get_train_data():
    return _get_data('Training')


def get_emotion_from_mapping(emotion_code):
    code_to_emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
    return code_to_emotion[emotion_code]


def get_sample():
    return _get_data(rows_to_read=1)
    
