'''
Created on Dec 13, 2017

Assumes fer2013 dataset is in ../data/fer2013

@author: Vladimir Petkov
'''

import pandas as pd
import numpy as np

def flat_to_matrix(pixels_string, size=(48,48)):
    pixels_arr = np.array(map(int, pixels_string.split()))
    return pixels_arr.reshape(size)

def _get_data(data_type):
    df=pd.read_csv('../data/fer2013.csv', sep=',')

    df_test = df[df.Usage == data_type] # to filter test data
    df_test['pixels'] = df_test.pixels.apply(lambda x: flat_to_matrix(x))

    x_test = np.array([matrix for matrix in df_test.pixels])
    y_test = np.array([emotion for emotion in df_test.emotion])
    
    return x_test, y_test

def get_test_data():
    return _get_data('PublicTest')

def get_train_data():
    return _get_data('Training')

def get_emotion_mapping():
    return {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
