
import tensorflow as tf
import numpy
from data import cifar10
from tensorflow.python.keras.layers import Input, Convolution2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.python.keras import utils, optimizers
import os

IMAGE_SIZE = 32
CLASSES_SIZE = 10 # Total number of classifications

# Hyper Parameters
LEARNING_RATE = 0.1
BATCH_SIZE = 10 # number of training examples used during a single epoch
EPOCH_TRAIN = 1 # number of iterations the training algorithim will do over the entire training set
EPOCH_EVAL = 20 # number of iterations for testing
FILTER_SIZE = 3 # 3X3 convolution layer filter(kernal) size
FCL_SIZE  = 512 # number of nodes in fully conneted hidden layer
CONV1_DEPTH = 32 # 32 filters (kernals) per conv layer
CONV2_DEPTH = 64 # using 64 filters per conv layer after 1st maxpooling
DROPOUT_1 = 0.25 # dropout probability just after last pooling
DROPOUT_2 = 0.5 # dropout probability in fully connected layer

# CIFAR-10 has 5000 training examples and 1000 tests examples
# Load Training Data
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data() 
print('x_train shape:', train_features.shape)
print('testing samples', test_features.shape[0])

test_tot = test_features.shape[0]
classes_tot = numpy.unique(train_labels).shape[0] # total number of classifications

# Normalize data
train_features = train_features.astype('float32')
train_features /= 255.0

# Convert labels to One-hot encoding 
train_labels = utils.to_categorical(train_labels, classes_tot)

##
# The Convolution Neural Network (CNN) is 4 layered. Each layer has 2D-Convolution,
# with the last two layers implementing 2D-Maxpooling for downsampling after Conv.
# Two fully Conected layers are implemented at the end of the network.
##
inferenceModel = tf.keras.Sequential() # Builds linear stack of layers for model

# Layer 1 - Conv(kernal size[32]) -> Relu 
inferenceModel.add(Convolution2D(CONV1_DEPTH, FILTER_SIZE,
	input_shape=train_features.shape[1:], padding='same'))
inferenceModel.add(Activation('relu'))

# Layer 2 - Conv(kernal size[32]) -> Relu -> Pool2D -> Dropout prob 0.25
inferenceModel.add(Convolution2D(CONV1_DEPTH, FILTER_SIZE, padding='same'))
inferenceModel.add(Activation('relu'))
inferenceModel.add(MaxPooling2D(pool_size=(2,2)))
inferenceModel.add(Dropout(DROPOUT_1))

# Layer 3 - Conv(kernal size[64]) -> Relu 
inferenceModel.add(Convolution2D(CONV2_DEPTH, FILTER_SIZE, padding='same'))
inferenceModel.add(Activation('relu'))

# Layer 4 - Conv(kernal size[32]) -> Relu -> Pool2D -> Dropout prob 0.5
inferenceModel.add(Convolution2D(CONV2_DEPTH, FILTER_SIZE, padding='same'))
inferenceModel.add(Activation('relu'))
inferenceModel.add(MaxPooling2D(pool_size=(2,2)))
inferenceModel.add(Dropout(DROPOUT_1))

# FCL 1 - Flatten to 1D -> Hidden FCL -> Relu -> Dropout prob 0.5
inferenceModel.add(Flatten()) 
inferenceModel.add(Dense(FCL_SIZE))
inferenceModel.add(Activation('relu'))
inferenceModel.add(Dropout(DROPOUT_2))

# FCL 2 - Flatten to 1D -> Final FCL -> softmax 
inferenceModel.add(Dense(classes_tot)) 
inferenceModel.add(Activation('softmax')) # produces error as a probability 

# Configure training process for Multi-class classification
optimizer = optimizers.Adam(lr=LEARNING_RATE)
inferenceModel.compile(
	optimizer=optimizer, # Adam optimizer or rmsprop
	loss='categorical_crossentropy', # use cros-entropy loss function to minimise loss
	metrics=['accuracy']) #  report on accuracy

# Model training
# Keep 10% of training data for validation
# Shuffle the batch before each epoch
# Steps_Per_EPOCH = dataset/BATCH_SIZE
# = Total number of (batches of samples) before declaring one epoch finished  
try:
	inferenceModel.fit(x=train_features, y=train_labels, 
		batch_size=BATCH_SIZE, epochs=EPOCH_TRAIN, 
		validation_split=0.1, shuffle=True, verbose=1) 
except (KeyboardInterrupt, SystemExit):
	# Allow exit of training 
	inferenceModel.save('savedh5/cnn_model.h5')
	print('Model saved...')
	raise
inferenceModel.save('savedh5/cnn_model.h5')
