import tensorflow as tf
from tensorflow.python.keras.layers import Input, Convolution2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import utils, optimizers, callbacks
import os, numpy as np
from keras.applications.vgg16 import VGG16 

# The following Convolution Neural Network (CNN) is trained to classifies up to
# 4 different vegetables. Classes include: Cabbage, Broccoli, Tomato and Onion.
# With further training, the model would be capable of recognizing more classes

# Hyper Parameters
IMAGE_SIZE = 48 # Dimensions of loaded image
CLASSES = 6 # Number of classifications (types of vegetables)
BATCH_SIZE = 16 # number of training examples used during a single epoch

# Directory Paths
main_path = 'C:\\Users\\jjmiy_000\\Documents\\GitHub' # root directory
dataset_path = os.path.join(main_path, 'data_sets') # Data set directory location
train_path = os.path.join(os.getcwd(), 'bottleneck_train_features.npy')
validate_path = os.path.join(os.getcwd(), 'bottleneck_validate_features.npy')

# Input Shape
shape = (IMAGE_SIZE, IMAGE_SIZE, 3) # input has form (height, width, channels)

vgg16_model = VGG16(input_shape=shape, weights='imagenet', include_top=False, pooling=max)
# TRAINGING AND VALIDATION DATASET  
# generates batches of normalized/augmented image training data in tensor form
data_gen = ImageDataGenerator(
	    rescale=1.0/255,
            data_format='channels_last',
            validation_split=0.1)

train_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=None,
        subset='training',
        shuffle=False)

validate_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=None,
        subset='validation',
        shuffle=False)

print(" Extract bottleneck features...")
# Get bottle neck features in numpay arrays from convolution part of model
bottleneck_train_features = vgg16_model.predict_generator(train_generator, len(train_generator),
                                                              verbose=1)
bottleneck_validate_features = vgg16_model.predict_generator(validate_generator, len(validate_generator),
                                                                 verbose=1)

# save the output of both training and validation as a Numpy array
np.save('bottleneck_train_features_1.npy', bottleneck_train_features)
np.save('bottleneck_validate_features_1.npy', bottleneck_validate_features)

