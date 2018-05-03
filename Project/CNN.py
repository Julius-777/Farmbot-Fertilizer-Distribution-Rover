
import tensorflow as tf
import numpy
from data import cifar10
from tensorflow.python.keras.layers import Input, Convolution2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.python.keras import utils, optimizers
import os
import threading

# # #
# The following Convolution Neural Network (CNN) is trained to classifies up to
# 4 different vegetables. Classes include: Cabbage, Broccoli, Tomato and Onion.
# With further training, the model would be capable of recognizing more classes
# # #

# Hyper Parameters
IMAGE_SIZE = 150 # Dimensions of loaded image
LEARNING_RATE = 0.1
BATCH_SIZE = 32 # number of training examples used during a single epoch
EPOCH_TRAIN = 10 # number of iterations the training algorithim will do over the entire training set
EPOCH_EVAL = 5 # number of iterations for testing
FILTER_SIZE = 3 # 3X3 convolution layer filter(kernal) size
FCL_SIZE  = 512 # number of nodes in fully conneted hidden layer
CONV1_DEPTH = 32 # 32 filters (kernals) per conv layer
CONV2_DEPTH = 64 # using 64 filters per conv layer after 1st maxpooling
DROPOUT_1 = 0.25 # dropout probability just after last pooling
DROPOUT_2 = 0.5 # dropout probability in fully connected layer

dataset_path = os.path.join(os.getcwd(), 'data_set') # Data set directory location

# The CNN is 4 layered. Each layer has 2D-Convolution,
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

# Configure image generator parameters 
data_gen = ImageDataGenerator(
	rescale=1.0/255, # rescale RGB inputs between 0-1
	shear_range=0.2,
	zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1) # 10% data reserved for validation testing

# Testing dataset
# generates batches of normalized/augmented image training data in tensor form
training_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        save_to_dir='/data',
        subset='training')

# Validation dataset
validation_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical', # multi-classification labels
        subset='validation') # Use validation data (10 %)

# Train and Validate the model
# Steps_Per_EPOCH = dataset/BATCH_SIZE
try:
	inferenceModel.fit_generator(dataset_generator, 
		batch_size=BATCH_SIZE,
		epochs=EPOCH_TRAIN,
		validation_data= validation_generator,
		verbose=1) 
	inferenceModel.save('savedh5/cnn_model.h5') #Save model parameters
	print('Model saved...')
except (KeyboardInterrupt, SystemExit):
	# Allow exit of training 
	inferenceModel.save('savedh5/cnn_model.h5')
	raise
