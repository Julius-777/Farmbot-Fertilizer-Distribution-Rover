
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Convolution2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import utils, optimizers, callbacks
import os

# # #
# The following Convolution Neural Network (CNN) is trained to classifies up to
# 4 different vegetables. Classes include: Cabbage, Broccoli, Tomato and Onion.
# With further training, the model would be capable of recognizing more classes
# # #

# Hyper Parameters
IMAGE_SIZE = 150 # Dimensions of loaded image
CLASSES = 4 # Number of classifications (types of vegetables)
LEARNING_RATE = 0.001 # rate at which gradient descent occurs
DECAY_RATE = 1e-6  # Learning rate decay per update
BATCH_SIZE = 32 # number of training examples used during a single epoch
BATCH_ITERATIONS = 90 # Number of batches iterated per epoch
EPOCHS_TRAIN = 50 # number of iterations the training algorithim will do over the entire training set
EVAL_ITERATIONS = 9 # number of iterations for testing
FILTER_SIZE = 3 # 3X3 convolution layer filter(kernal) size
FCL_SIZE  = 512 # number of nodes in fully conneted hidden layer
CONV1_DEPTH = 32 # 32 filters (kernals) per conv layer
CONV2_DEPTH = 64 # using 64 filters per conv layer after 1st maxpooling
DROPOUT_1 = 0.25 # dropout probability just after last pooling
DROPOUT_2 = 0.5 # dropout probability in fully connected layer

main_path = 'C:\\Users\\jjmiy_000\\Documents' # root directory
dataset_path = 'data_sets' # Data set directory location

# The CNN is 4 layered. Each layer has 2D-Convolution,
# with the last two layers implementing 2D-Maxpooling for downsampling after Conv.
# Two fully Conected layers are implemented at the end of the network.
##
inferenceModel = tf.keras.Sequential() # Builds linear stack of layers for model

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3) # input has form (samples, height, width, channels)
# Layer 1 - Conv(kernal_size=32) -> Relu
inferenceModel.add(Convolution2D(CONV1_DEPTH, FILTER_SIZE, input_shape=input_shape,
 								padding='same'))
inferenceModel.add(Activation('relu'))

# Layer 2 - Conv(kernal_size=32) -> Relu -> Pool2D -> Dropout prob 0.25
inferenceModel.add(Convolution2D(CONV1_DEPTH, FILTER_SIZE, padding='same'))
inferenceModel.add(Activation('relu'))
inferenceModel.add(MaxPooling2D(pool_size=(2,2)))
inferenceModel.add(Dropout(DROPOUT_1))

# Layer 3 - Conv(kernal_size=64) -> Relu
inferenceModel.add(Convolution2D(CONV2_DEPTH, FILTER_SIZE, padding='same'))
inferenceModel.add(Activation('relu'))

# Layer 4 - Conv(kernal_size=64) -> Relu -> Pool2D -> Dropout prob 0.5
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
inferenceModel.add(Dense(CLASSES))
inferenceModel.add(Activation('sigmoid')) # produces error as a probability

optimizer = optimizers.Adam(lr=LEARNING_RATE, decay=DECAY_RATE)
# Configure training process for Multi-class classification
inferenceModel.compile(
	optimizer=optimizer, # Adam optimizer or rmsprop
	loss='categorical_crossentropy', # use cros-entropy loss function to minimise loss
	metrics=['accuracy']) #  report on accuracy

# Configure parameters for preparing the data set as an input
data_gen = ImageDataGenerator(
	rescale=1.0/255, # rescale RGB inputs between 0-1
	shear_range=0.2,
	zoom_range=0.2,
    horizontal_flip=True,
	data_format='channels_last',
    validation_split=0.1) # 10% data reserved for validation testing

# Batch generator for TRAINGING
# generates batches of normalized/augmented image training data in tensor form
training_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training')

# Batch generator for TESTING
validation_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical', # multi-classification labels
        subset='validation') # Use validation data (10 %)

# Allows visualization of training progress on tensorboard
visualize = callbacks.TensorBoard(log_dir='./graph_log2', histogram_freq=0,
          write_graph=True, write_images=True)

# Train and Evaluate the model
# Steps_Per_EPOCH = dataset/BATCH_SIZE
try:
	inferenceModel.fit_generator(training_generator,
		steps_per_epoch=BATCH_ITERATIONS, # Number of batches per epoch
		epochs=EPOCHS_TRAIN,
        validation_steps=EVAL_ITERATIONS,
		validation_data= validation_generator,
        callbacks=[visualize],
		verbose=1)
	inferenceModel.save('savedh5/cnn_model.h5') #Save model parameters
	print('Model saved...')
except (KeyboardInterrupt, SystemExit):
	# Allow exit of training
	inferenceModel.save('savedh5/cnn_model.h5')
	print('Emergency saved...')
	raise
# Test using:
# eval = inferenceModel.evaluate_generator(validation_generator)
# print("loss: ", eval[0], " accurarcy: ",eval[1])
