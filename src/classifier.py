'''
Created on Dec 17, 2017

TODO

@author: Vladimir Petkov
'''

from keras.models import model_from_json
from src import data_preparator
from src.data_preparator import get_emotion_from_mapping
import numpy as np


class Classifier(object):
    """
    TODO
    """

    def __init__(self):
        self.model = self._load_model()

    def _load_model(self, models_path="../models/"):
        print "Loading model from disk"
        with open(models_path + "model.json", "r") as json_file: 
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(models_path + "model.h5")
        
        return model

    def classify(self, image):
        """
        TODO
        """
        prediction = self.model.predict(image, verbose=1)
        prediction_index = -1
        for class_result in np.nditer(prediction):
            prediction_index += 1
            if class_result == 1:
                break
        return get_emotion_from_mapping(prediction_index)
        

def main():  # for test purposes
    sample_x, sample_y = data_preparator.get_sample()
    classifier = Classifier()
    prediction = classifier.classify(sample_x)
    print 'Actual type: ', get_emotion_from_mapping(np.take(sample_y, 0))
    print 'Predicted type: ', prediction


if __name__ == "__main__":
    main()
