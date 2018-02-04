'''
Created on Dec 17, 2017

TODO

@author: Vladimir Petkov
'''

from os import getenv, path
from keras.models import model_from_json
from data_loader import get_emotion_from_code
import numpy as np

MODEL_DIR = getenv("MODEL_DIR", "../models")
MODEL_WEIGHTS_FILENAME = getenv("MODEL_WEIGHTS_FILENAME", "model.h5")
MODEL_CONFIG_FILENAME = getenv("MODEL_CONFIG_FILENAME", "model.json")


class EmotionClassifier(object):
    """
    Finds emotion from a picture of a face.
    """

    def __init__(self):
        self.model = self._load_model()

    def _load_model(self, models_path=MODEL_DIR):
        print "Loading model from disk"
        with open(path.join(models_path, MODEL_CONFIG_FILENAME), "r") as json_file: 
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(path.join(models_path, MODEL_WEIGHTS_FILENAME))
        return model

    def classify(self, image):
        """
        Returns one of the seven universal facial expressions: happy, sad, angry, fear, disgust, neutral, surprise
        
        Args:
            image (numpy array): grayscale image in size 48x48
        Returns:
            string: one of the seven universal facial expressions.
        """
        image = image.reshape(-1, image.shape[0], image.shape[1], 1)
        prediction = self.model.predict(image, verbose=1)
        max_index = np.argmax(prediction)
        return get_emotion_from_code(max_index)
