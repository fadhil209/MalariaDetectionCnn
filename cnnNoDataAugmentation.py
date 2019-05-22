import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

import keras
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
from keras import backend as K
from keras import optimizers
from keras.utils import np_utils as keras_utils

print(os.listdir("cell_images"))
print("\n")


parasitized_data = os.listdir("cell_images/Parasitized")
print(parasitized_data[:10])
print("\n")

uninfected_data = os.listdir("cell_images/Uninfected")
print(uninfected_data[:10])


# Data Visuallization

# Infected Data
plt.figure(figsize=(12, 12))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    img = cv2.imread("cell_images/Parasitized/" + parasitized_data[i])
    plt.imshow(img)
    plt.title("PARASITIZED : 1")
    plt.tight_layout()
plt.show()

# Uninfected Data
plt.figure(figsize=(12, 12))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    img = cv2.imread("cell_images/Uninfected/" + uninfected_data[i])
    plt.imshow(img)
    plt.title("UNINFECTED : 0")
    plt.tight_layout()
plt.show()

data_train = []
labels_train = []
data_test = []
labels_test = []

# # append all the images in one data variable and append the respective labels in labels variable
# for img in parasitized_data:
#     try:
#         img_read = plt.imread("cell_images/Parasitized/" + img)
#         img_resize = cv2.resize(img_read, (50, 50))
#         img_array = img_to_array(img_resize)
#         data.append(img_array)
#         labels.append(1)
#     except:
#         None
# for img in uninfected_data:
#     try:
#         img_read = plt.imread("cell_images/Uninfected/" + img)
#         img_resize = cv2.resize(img_read, (50, 50))
#         img_array = img_to_array(img_resize)
#         data.append(img_array)
#         labels.append(0)
#     except:
#         None

# plt.imshow(data[15000])
# plt.title(labels[15000])
# plt.show()


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


# print(data_train[4])

# for i in range(4):
#     plt.subplot(1, 4, i + 1)
#     # img = cv2.imread(data_train[i])
#     plt.imshow(data_train[i])
#     plt.title("label : " + str(labels_train[i]))
#     plt.tight_layout()
# plt.show()

# print("train Data: " + str(len(data_train)) + "," + str(len(labels_train)))
# print("test Data: " + str(len(data_test)) + "," + str(len(labels_test)))
# print("total data" + str(len(parasitized_data) + len(uninfected_data)))
# print("Total data to work with" + str(len(data_train) + len(data_test)))

# print(f'SHAPE OF TRAINING IMAGE DATA : {np.shape(data_train)}')
# print(f'SHAPE OF TESTING IMAGE DATA : {np.shape(data_test)}')


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


model = CNN(50, 50, 2, 3)
model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

data_train, labels_train, data_test, labels_test = trainAndTest(parasitized_data, uninfected_data, 1)
print(labels_test[0])
labels_train = keras_utils.to_categorical(labels_train, num_classes=2)
labels_test = keras_utils.to_categorical(labels_test, num_classes=2)

print(f'SHAPE OF TRAINING IMAGE DATA : {np.array(data_train).shape}')
print(f'SHAPE OF TRAINING LABELS : {np.array(labels_train).shape}')

model.fit(np.array(data_train), np.array(labels_train), epochs=1, batch_size=32)

# print(model.predict(np.array(data_test[:4])))
# # model.predict(np.array(data_test[:1]))
# print(labels_test[0])
# print(labels_test[1])
# print(labels_test[2])
# print(labels_test[3])

predictions = model.evaluate(np.array(data_test), labels_test)

print(f'LOSS : {predictions[0]}')
print(f'ACCUACY : {predictions[1]}')
