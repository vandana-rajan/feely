'''
Created on Feb 4, 2018

@author: Vladimir Petkov
'''

'''
load image 2 emotion mapping csv
load image from file
crop face
to grayscale
export to csv
'''

from os import getenv
import glob
import numpy as np
import cv2
import pandas as pd
from keras.utils import to_categorical
from PIL import Image

FRONTAL_FACE_XML = getenv("FRONTAL_FACE_XML", "../resources/haarcascade_frontalface_default.xml")

def get_face(image_path):
    image = Image.open(image_path)
    return cv2.resize(np.array(image), (48, 48))
    image_array = np.array(image)
    grayscale_image = image_array
    faceCascade = cv2.CascadeClassifier(FRONTAL_FACE_XML)
#     grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    face_coordinates = faceCascade.detectMultiScale(grayscale_image, scaleFactor=1.001, minNeighbors=5, minSize=(48, 48))
    for (x, y, w, h) in face_coordinates:
        cropped_image = grayscale_image[y: y + h, x: x + w]
        return cv2.resize(cropped_image, (48, 48))

    
def jaffe_get_data(dataset_path='../../datasets/jaffe'):
    x = []
    y = []
    for neutral in glob.glob(dataset_path + '/*NE*tiff'):
        x.append(get_face(neutral))
        y.append(6)
    for happy in glob.glob(dataset_path + '/*HA*tiff'):
        x.append(get_face(happy))
        y.append(3)
    for sad in glob.glob(dataset_path + '/*SA*tiff'):
        x.append(get_face(sad))
        y.append(4)
    for surprise in glob.glob(dataset_path + '/*SU*tiff'):
        x.append(get_face(surprise))
        y.append(5)
    for angry in glob.glob(dataset_path + '/*AN*tiff'):
        x.append(get_face(angry))
        y.append(0)
    for disgust in glob.glob(dataset_path + '/*DI*tiff'):
        x.append(get_face(disgust))
        y.append(1)
    for fear in glob.glob(dataset_path + '/*FE*tiff'):
        x.append(get_face(fear))
        y.append(2)
    x_np = np.array(x)
    y_np = np.array(y)
    x_np = np.reshape(x_np, (len(x), 48, 48, 1))
    y_np = to_categorical(y_np)
    return x, y
        


def flat_to_matrix(pixels_string, size=(48, 48)):
    pixels_arr = np.array(map(int, pixels_string.split()))
    return pixels_arr.reshape(size)


def fer2013_get_data(data_type=None, dataset_path='../../datasets/fer2013'):
    df = pd.read_csv(dataset_path + '/fer2013.csv', sep=',')

    if data_type is not None:
        df = df[df.Usage == data_type]
    df['pixels'] = df.pixels.apply(lambda x: flat_to_matrix(x))

    x = np.array([matrix for matrix in df.pixels])
    y = np.array([emotion for emotion in df.emotion])
    
    x = x.reshape(-1, x.shape[1], x.shape[2], 1)
    y = to_categorical(y)
    
    return x, y

def affectnet_get_data(data_type, dataset_path='../../datasets/affectnet'):
    df = pd.read_csv(dataset_path + '/' + data_type, sep=',')

    df['pixels'] = df.pixels.apply(lambda x: flat_to_matrix(x))
    x = np.array([matrix for matrix in df.pixels])
    y = np.array([emotion for emotion in df.emotion])
    
    x = x.reshape(-1, x.shape[1], x.shape[2], 1)
    y = to_categorical(y)
    
    return x, y

def save_array(array_tuple, dataset, usage_type):
    x, y = array_tuple
    target_path = '../data'
    np.save(target_path + '/' + dataset + '/' + usage_type + '_x', x)
    np.save(target_path + '/' + dataset + '/' + usage_type + '_y', y)


def main(dataset='affectnet'):
    
    if dataset == 'fer2013':
        save_array(fer2013_get_data('Training'), dataset, 'train')
        save_array(fer2013_get_data('PublicTest'), dataset, 'validation')
        save_array(fer2013_get_data('PrivateTest'), dataset, 'test')
    elif dataset == 'jaffe':
        save_array(jaffe_get_data(), dataset, 'test') # really low amount of data so only test
    elif dataset == 'affectnet':
        save_array(affectnet_get_data('training.csv'), dataset, 'train')
        #save_array(affectnet_get_data(), dataset, 'validation')
    

if __name__ == "__main__":
    main()
