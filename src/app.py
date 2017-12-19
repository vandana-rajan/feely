'''
Created on Dec 17, 2017

Application for emotion recognition from photos.

@author: Vladimir Petkov
'''

import cv2
from os import getenv
from src.emotion_classifier import EmotionClassifier

FRONTAL_FACE_XML = getenv("FRONTAL_FACE_XML", "../resources/haarcascade_frontalface_default.xml")


def take_picture():
    """
    Takes picture from camera on port 0.
    
    Returns:
        numpy array: the image taken as numpy array
    """
    camera = cv2.VideoCapture(0)
    _, image = camera.read()
    del(camera)
    return image


def find_faces(image):
    """
    Finds faces in the provided picture.
    
    Args:
        image (numpy array): the image to be processed
    Returns:
        array of numpy arrays: list of 48x48 images containing the faces found in the picture
    """
    faceCascade = cv2.CascadeClassifier(FRONTAL_FACE_XML)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_coordinates = faceCascade.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    faces = []
    for (x, y, w, h) in face_coordinates:
        cropped_image = grayscale_image[y: y + h, x: x + w]
        resized_image = cv2.resize(cropped_image, (48, 48))
        faces.append(resized_image)
    return faces


def main():
    classifier = EmotionClassifier()
    image = take_picture()
    faces = find_faces(image)
    for face in faces:
        emotion = classifier.classify(face)
        print emotion
        cv2.imshow(emotion, face)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
