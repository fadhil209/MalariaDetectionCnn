import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

from keras.preprocessing.image import img_to_array

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

data = []
labels = []

for img in parasitized_data:
    try:
        img_read = plt.imread("cell_images/Parasitized/" + img)
        img_resize = cv2.resize(img_read, (50, 50))
        img_array = img_to_array(img_resize)
        data.append(img_array)
        labels.append(1)
    except:
        None
for img in uninfected_data:
    try:
        img_read = plt.imread("cell_images/Uninfected/" + img)
        img_resize = cv2.resize(img_read, (50, 50))
        img_array = img_to_array(img_resize)
        data.append(img_array)
        labels.append(0)
    except:
        None

plt.imshow(data[15000])
plt.title(labels[15000])
plt.show()
