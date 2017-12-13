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

print 'Loading data'
x_test, y_test = get_test_data()
x_train, y_train = get_train_data()

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

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

print 'Starting training'
model.fit(x_test, y_test, batch_size=32, epochs=2, verbose=2) # TODO: switch to train data

print 'Evaluating'
score = model.evaluate(x_test, y_test, batch_size=32)
print score