'''
Created on Dec 12, 2017

TODO:

@author: Vladimir Petkov
'''

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from data_preparator import get_test_data
from data_preparator import get_train_data


def create_model():
    print 'Creating model'
    model = Sequential()
    
    # input: 48x48 images with 1 channel
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model


def save_model(model, save_path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(save_path + "model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(save_path + "model.h5")
    print("Saved model to disk")


def main():
    model = create_model()
    
    print 'Loading data'
    x_test, y_test = get_test_data()
    x_train, y_train = get_train_data()

    print 'Starting training'
    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
    
    print 'Evaluating'
    loss_and_accuracy = model.evaluate(x_test, y_test, batch_size=32)
    print "Loss: ", loss_and_accuracy[0]
    print "Accuracy: ", loss_and_accuracy[1]
    
    print 'Saving model'
    save_model(model, "../models/")
    

if __name__ == "__main__":
    main()
