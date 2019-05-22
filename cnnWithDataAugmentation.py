import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

import keras
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
from keras import backend as K
from keras import optimizers
from keras.utils import np_utils as keras_utils
import time

from keras.preprocessing.image import ImageDataGenerator


"""
divides the images into Train and Test parts when 1 is given as testPart there
the 1st quarter of the images will be Test and the rest will be trained and the
same if 2 is given as testPart then the 2nd Quarter of the images will be Test
data and the rest as Train data

keyword arguments:
parasitizedData -- array or list of infected images
uninfectedData -- array or list of uninfected images

testPart -- int number of which quarter of image data to be Test data

This function is there because as discussed with professor we shall have the test
data as the first quarter of the images in the first run and the last 3 quarters of
images data will be the training Data and the second run the 2nd quarter
of the images will be left as Test and the 1st, 3rd and 4th quarters will be left
for training and with this function we can divide according to what needs to be
Test data and what needs to be Train Data
"""


def trainAndTest(parasitizedData, uninfectedData, testPart):

    testData = []
    testLabel = []
    trainData = []
    trainLabel = []
    onePart = len(parasitizedData) / 4
    testEndIndex = onePart * testPart
    testStartIndex = testEndIndex - onePart

    for i in range(len(parasitizedData)):
        if (testStartIndex <= i and i <= testEndIndex):
            try:
                img_read = plt.imread("cell_images/Parasitized/" + parasitizedData[i])
                img_resize = cv2.resize(img_read, (50, 50))
                img_array = img_to_array(img_resize)
                testData.append(img_array)
                testLabel.append(1)
                img_read = plt.imread("cell_images/Uninfected/" + uninfectedData[i])
                img_resize = cv2.resize(img_read, (50, 50))
                img_array = img_to_array(img_resize)
                testData.append(img_array)
                testLabel.append(0)
            except:
                None
        else:
            try:
                img_read = plt.imread("cell_images/Parasitized/" + parasitizedData[i])
                img_resize = cv2.resize(img_read, (50, 50))
                img_array = img_to_array(img_resize)
                trainData.append(img_array)
                trainLabel.append(1)
                img_read = plt.imread("cell_images/Uninfected/" + uninfectedData[i])
                img_resize = cv2.resize(img_read, (50, 50))
                img_array = img_to_array(img_resize)
                trainData.append(img_array)
                trainLabel.append(0)
            except:
                None
    return trainData, trainLabel, testData, testLabel


"""
Build a CNN model that can be used in the main program to train, test and predict also

Keyword arguments:
height -- the height of the images
width -- the width of the images
classes -- the number of different classes
channels -- the number of channels which can be known from numpy array .shape() method
"""


def CNN(height, width, classes, channels):
    model = Sequential()

    inputShape = (height, width, channels)

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=inputShape, data_format='channels_last'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    return model


"""
Main Program
"""


def main():
    parasitized_data = os.listdir("cell_images/Parasitized")
    uninfected_data = os.listdir("cell_images/Uninfected")

    train_datagenerator = ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest',
        zoom_range=0.3,
        rotation_range=30)

    for i in range(4):
        data_train = []
        data_test = []
        labels_train = []
        labels_test = []

        print('Dividing Data into Train and Test sets')
        data_train, labels_train, data_test, labels_test = trainAndTest(parasitized_data, uninfected_data, i + 1)
        print('Done dividing will start training now')
        model = CNN(50, 50, 2, 3)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        labels_train = keras_utils.to_categorical(labels_train, num_classes=2)
        labels_test = keras_utils.to_categorical(labels_test, num_classes=2)

        model.fit_generator(train_datagenerator.flow(np.array(data_train), labels_train, batch_size=32, shuffle=False),
                            steps_per_epoch=len(data_train) // 32, epochs=3)
        print('finished training')

        # prediction = model.predict_generator(test_datagen.flow(np.array(data_test[:4])), steps=5)
        prediction = model.predict(np.array(data_test[:4]))
        plt.figure(figsize=(10, 10))
        for j in range(4):
            plt.subplot(1, 4, j + 1)
            plt.imshow(data_test[j])
            if(prediction[j][0] > prediction[j][1]):
                predicted = 'Uninfected'
            else:
                predicted = 'Parasitized'
            if(labels_test[j][0] > labels_test[j][1]):
                actual = 'Uninfected'
            else:
                actual = 'Parasitized'
            title = 'actual : ' + actual + '\npredicted : ' + predicted
            plt.title(title)
            plt.tight_layout()
        plt.show()

        predictions = model.evaluate(np.array(data_test), labels_test)

        print(f'with {i + 1} quarter of images as Test Data')
        print(f'LOSS : {predictions[0]}')
        print(f'ACCURACY : {predictions[1]}')
        time.sleep(5)



# Running the main program only
main()
