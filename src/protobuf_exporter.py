'''
Created on Dec 24, 2017

@author: Vladimir Petkov
'''

from os import getenv, path
from keras.models import model_from_json
from keras import backend as K
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
import tensorflow as tf

# from src.data_preparator import get_emotion_from_code
# import numpy as np

MODEL_DIR = getenv("MODEL_DIR", "../models")
MODEL_WEIGHTS_FILENAME = getenv("MODEL_WEIGHTS_FILENAME", "model.h5")
MODEL_CONFIG_FILENAME = getenv("MODEL_CONFIG_FILENAME", "model.json")

MODEL_NAME = "model_pb"

model = None


def load_model(models_path=MODEL_DIR):
        print "Loading model from disk"
        with open(path.join(models_path, MODEL_CONFIG_FILENAME), "r") as json_file: 
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(path.join(models_path, MODEL_WEIGHTS_FILENAME))
        return model


def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
                         MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
                              False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
                              "save/restore_all", "save/Const:0", \
                              'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")
    
    
def main():
    K.set_learning_phase(0);
    model = load_model()
    export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_3/Softmax")


if __name__ == "__main__":
    main()
