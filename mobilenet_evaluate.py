import CNN
from tensorflow.python.keras.layers import Input, Convolution2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import utils, optimizers, callbacks
import os, numpy as np
from keras.applications.mobilenet import MobileNet

saved_path = os.path.join(os.getcwd(), 'savedh5/mobilenet_top_layer_weights.h5')
# Input Shape
shape = (128, 128, 3) # input has form (height, width, channels)

# LOAD MOBILENET pre-trained model
def load_mobile_net():
    return MobileNet(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
                  alpha=1,
                  include_top=False,
                  weights='imagenet',
                  input_tensor=None,
                  pooling=max)


def load_top_layer(bottleneck):
    top_layer = CNN.new_top_layer(bottleneck)
    top_layer.load_weights(saved_path)
    return top_layer

def get_prediction(image_array, model, top_layer):
    bottleneck_prediction = model.predict(image_array)
    prediction = top_layer.predict_classes(bottleneck_prediction)
    return prediction
