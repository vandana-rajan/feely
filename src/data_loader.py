'''
Created on Dec 13, 2017

Loads dataset as numpy arrays prepraed for training

Assumes there are prepared numpy arrays in the path provided by DATA_PATH variable 

@author: Vladimir Petkov
'''

from os import getenv
import numpy as np

DATA_PATH = getenv('DATA_PATH', '../data')

def get_test_data(dataset='fer2013'):
    x = np.load('/'.join((DATA_PATH, dataset, 'test_x.npy')))
    y = np.load('/'.join((DATA_PATH, dataset, 'test_y.npy')))
    return x, y


def get_validation_data(dataset='fer2013'):
    x = np.load('/'.join((DATA_PATH, dataset, 'validation_x.npy')))
    y = np.load('/'.join((DATA_PATH, dataset, 'validation_y.npy')))
    return x, y


def get_train_data(dataset='fer2013'):
    x = np.load('/'.join((DATA_PATH, dataset, 'train_x.npy')))
    y = np.load('/'.join((DATA_PATH, dataset, 'train_y.npy')))
    return x, y


def get_emotion_from_code(emotion_code):
    code_to_emotion = {0:'Anger', 1:'Disgust', 2:'Fear', 3:'Happiness', 4:'Sadness', 5:'Surprise', 6:'Neutral'}
    return code_to_emotion[emotion_code]
