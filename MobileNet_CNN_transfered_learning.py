import tensorflow as tf
from tensorflow.python.keras.layers import Input, Convolution2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import utils, optimizers, callbacks
import os, numpy as np
from keras.applications.mobilenet import MobileNet

# Hyper Parameters
IMAGE_SIZE = 128 # Dimensions of loaded image
CLASSES = 7 # Number of classifications (types of vegetables)
LEARNING_RATE = 0.0001 # rate at which gradient descent occurs
DECAY_RATE = 1e-6  # Learning rate decay per update
BATCH_SIZE = 32 # number of training examples used during a single epoch
EPOCHS_TRAIN = 50 # number of iterations the training algorithim will do over the entire training set
FILTER_SIZE = 3 # 3X3 convolution layer filter(kernal) size
FCL_SIZE  = 128 # number of nodes in fully conneted hidden layer
CONV1_DEPTH = 32 # 32 filters (kernals) per conv layer
CONV2_DEPTH = 64 # using 64 filters per conv layer after 1st maxpooling
DROPOUT_2 = 0.5 # dropout probability in fully connected layer

# Directory Paths
main_path = 'C:\\Users\\jjmiy_000\\Documents\\GitHub' # root directory
dataset_path = os.path.join(main_path, 'data_sets') # Data set directory location
train_path = os.path.join(os.getcwd(), 'bottleneck_train_features.npy')
validate_path = os.path.join(os.getcwd(), 'bottleneck_validate_features.npy')

# Input Shape
shape = (IMAGE_SIZE, IMAGE_SIZE, 3) # input has form (height, width, channels)

# LOAD MOBILENET pre-trained model
mobileNet_model = MobileNet(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
              alpha=1,
              include_top=False,
              weights='imagenet',
              input_tensor=None,
              pooling=max)

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
        class_mode='categorical',
        subset='training',
        shuffle=False)

validate_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False)

print(" Extract bottleneck features...")
# Get bottle neck features in numpay arrays from convolution part of model
bottleneck_train_features = mobileNet_model.predict_generator(train_generator,
                                                              len(train_generator),verbose=1)
bottleneck_validate_features = mobileNet_model.predict_generator(validate_generator,
                                                                 len(validate_generator), verbose=1)

# Get training labels
print(" Get Labels...")
num_classes = len(train_generator.class_indices)
train_classes = train_generator.classes  
train_labels = utils.to_categorical(train_classes, num_classes=num_classes)

num_classes = len(validate_generator.class_indices)
validate_classes = validate_generator.classes  
validate_labels = utils.to_categorical(validate_classes, num_classes=num_classes)

print("Create top layer model...")
# Builds linear stack of layers for model
model = tf.keras.Sequential()
# FCL 1 - Flatten to 1D -> Hidden FCL -> Relu -> Dropout prob 0.5
model.add(Flatten(input_shape=bottleneck_train_features.shape[1:]))
model.add(Dense(FCL_SIZE))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT_2))
# FCL 2 - Flatten to 1D -> Final Fully Connected Layer -> softmax
model.add(Dense(CLASSES))
model.add(Activation('softmax')) # produces error as a probability

# Configure optimizer for gradient descent
# Compile training process for Multi-class classification
optimizer = optimizers.RMSprop(lr=LEARNING_RATE, decay=DECAY_RATE)
model.compile(
	optimizer=optimizer, # Adam optimizer or rmsprop
	loss='categorical_crossentropy', # use cros-entropy loss function to minimise loss
	metrics=['accuracy']) #  report on accuracy

print(" Train top layer model on bottleneck features...")
# Train Top layer
try:
    model.fit(bottleneck_train_features, train_labels,
              epochs=EPOCHS_TRAIN,
              batch_size=BATCH_SIZE,
              verbose=0,
              validation_data=(bottleneck_validate_features, validate_labels))
    model.save('mobilenet_model2.h5') 
    model.save_weights('mobilenet_top_layer_weights3.h5')
except (KeyboardInterrupt, SystemExit):
    # Allow exit of training
    print('Emergency Exit...')
    model.save('mobilenet_model_.h5') 
    model.save_weights('mobilenet_top_layer_weights3.h5')
           
# Evaluate Model performance:
evaluate = model.evaluate(bottleneck_validate_features, validate_labels,
                          batch_size=BATCH_SIZE, verbose=1)
print("loss: ", evaluate[0], " accurarcy: ",evaluate[1])
