# -*- coding: utf-8 -*-
"""16FACE-THERMAL-VGG16-FINETUNING.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Lb_kygltTixocXNDZHrBbpkFLiSF3Icj

# implementation description

```
The thermal face recognition process is undertaken using convolutional neural networks (CNN's). More precisely, the system
comprises the first 10 layers from the VGG16 architecture, followed by a max-pooling layer, a batch-normalization layer
and a classifier (densely connected layer)
```
"""

from google.colab import drive
drive.mount('/content/drive/')

"""# import packages and libraries"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers, optimizers
from keras.models import Model
from keras import layers
from keras import models
from keras.layers import BatchNormalization
from tensorflow.keras.utils import Sequence

import os, shutil
import numpy as np
import time
from PIL import Image


from keras import metrics
import functools
from functools import partial

import matplotlib.pyplot as plt

"""# load VGG-16 model"""

# Load the VGG16 model
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(72, 96, 3))

conv_base.summary()

for i in range (1,10):
    conv_base.layers.pop()

conv_base.summary()

inp = conv_base.input
out =conv_base.layers[-1].output

thermalModel = Model(inp, out)

#From the first 10 layers, only the last three are set up to allow the training.
cont = 0
for layer in thermalModel.layers:
    cont = cont + 1
    if (cont >= 8):
        layer.trainable = True
    else:
        layer.trainable = False

thermalModel.summary()



"""# preprocess data"""

def count_directories(path):
    return len([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])

train_classes = count_directories('/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/train')
test_classes = count_directories('/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/test')
validation_classes = count_directories('/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/val')

print(f"Training classes: {train_classes}")
print(f"Testing classes: {test_classes}")
print(f"Validation classes: {validation_classes}")

# Directories for train, val, test
train_dir = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/train'
val_dir = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/val'
test_dir = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/test'

"""# modify model for fine-tuning"""

model = models.Sequential()
model.add(thermalModel)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='softmax'))

model.summary()

"""# compile model"""

top2_acc = functools.partial(metrics.top_k_categorical_accuracy, k=2)
top3_acc = functools.partial(metrics.top_k_categorical_accuracy, k=3)
top4_acc = functools.partial(metrics.top_k_categorical_accuracy, k=4)
top5_acc = functools.partial(metrics.top_k_categorical_accuracy, k=5)
top2_acc.__name__ = 'top2_acc'
top3_acc.__name__ = 'top3_acc'
top4_acc.__name__ = 'top4_acc'
top5_acc.__name__ = 'top5_acc'

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4), #Decrease learning rate
              metrics=['accuracy',top2_acc, top3_acc, top4_acc, top5_acc])

"""# image generator"""

# Image size and batch size
image_size = (72, 96)
batch_size = 4

# Data augmentation for training data
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# Only rescaling for training, validation, and test data
datagen = ImageDataGenerator(rescale=1./255)

# Only rescaling for validation and test data
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical'
)

# # Print the number of samples to ensure everything is set up correctly
# print(f"Found {train_generator.samples} images belonging to {train_generator.num_classes} classes for train")
# print(f"Found {val_generator.samples} images belonging to {val_generator.num_classes} classes for validation")
# print(f"Found {test_generator.samples} images belonging to {test_generator.num_classes} classes for test")

"""# remove corrupted images"""

# def plot_images(generator):
#     images, labels = next(generator)
#     plt.figure(figsize=(10, 10))
#     for i in range(len(images)):
#         plt.subplot(4, 4, i + 1)
#         plt.imshow(images[i])
#         plt.axis('off')
#     plt.show()

# plot_images(train_generator)

# def test_read_images(directory):
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith(('jpg', 'jpeg', 'png')):
#                 file_path = os.path.join(root, file)
#                 try:
#                     img = Image.open(file_path)
#                     img.load()
#                     print(f"Successfully loaded: {file_path}")
#                 except (IOError, SyntaxError) as e:
#                     print(f"Problem with image {file_path}: {e}")

# # Test the subsets
# test_read_images(train_dir)
# test_read_images(val_dir)
# test_read_images(test_dir)

# def find_and_remove_corrupted_images(directory):
#     """
#     Identify and remove corrupted images in the specified directory.

#     Parameters:
#     - directory: The directory to check for corrupted images.
#     """
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith(('jpg', 'jpeg', 'png')):
#                 file_path = os.path.join(root, file)
#                 try:
#                     img = Image.open(file_path)
#                     img.verify()  # Verify that it is, in fact, an image
#                 except (IOError, SyntaxError, Image.DecompressionBombError) as e:
#                     print(f"Corrupted image found and removed: {file_path}")
#                     os.remove(file_path)

# # Directories to check
# directories = [
#     '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH/data/ExtractedTerravicDatabase_subset/train',
#     '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH/data/ExtractedTerravicDatabase_subset/val',
#     '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH/data/ExtractedTerravicDatabase_subset/test'
# ]

# # Find and remove corrupted images
# for directory in directories:
#     find_and_remove_corrupted_images(directory)

# print("Corrupted images check and removal complete.")

"""# train model"""

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    workers=0,
    max_queue_size=0
)

"""# evaluate model"""

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test accuracy: {test_acc}')

"""# save the model"""

# model.save('thermal_face_recognition_vgg16.h5')