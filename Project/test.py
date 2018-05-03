
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Convolution2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.python.keras import utils
import os

IMAGE_SIZE = 32
CLASSES_SIZE = 10 
save_dir = os.path.join(os.getcwd(), 'saved')

# Hyper Parameters
BATCH_SIZE = 32 # number of training examples used during a single epoch
EPOCH_TRAIN = 100 # number of iterations the training algorithim will do over the entire training set
EPOCH_EVAL = 20 # number of iterations for testing
FILTER_SIZE = 3 # 3X3 convolution layer filter(kernal) size
FCL_SIZE  = 512 # number of nodes in fully conneted hidden layer
CONV1_DEPTH = 32 # 32 filters (kernals) per conv layer
CONV2_DEPTH = 64 # using 64 filters per conv layer after 1st maxpooling
DROPOUT_1 = 0.25 # dropout probability just after last pooling
DROPOUT_2 = 0.5 # dropout probability in fully connected layer

inferenceModel = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
inferenceModel.compile(
	optimizer='rmsprop', # Adam optimizer or rmsprop
	loss='categorical_crossentropy', # use cros-entropy loss function to minimise loss
	metrics=['accuracy'])

inferenceModel.save('savedh5/cnn_model.h5')
