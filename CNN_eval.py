#!/usr/bin/python -W ignore::DeprecationWarning
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import models
from tensorflow.python.keras.applications.mobilenet import MobileNet
from picamera.array import PiRGBArray
from picamera import PiCamera
import os, time, CNN
import numpy as np

IMAGE_SIZE = 128 # Dimensions of loaded image
main_path = '/Home/pi/Kilimanjaro' # root directory

# path for loading the saved top layer of model
save_path = os.path.join(os.getcwd(), 'savedh5/mobilenet_top_layer2.h5')

# all new operations will be in test mode from now on
K.set_learning_phase(0)

class PlantDetection:

    def __init__(self):
        # Dictionary holding classes with their corresponding labels
        self.class_dictionary = {0:"Broccoli", 1:"Cabbage", 2:"Purple Onion",
                                 3:"Spinach", 4:"Strawberry", 5:"Tomato", 6:"Yellow Onion"}
        # load top layer and convolution layers
        self.restoredModel = self.load_mobile_net(0.5)
        self.restoredTopLayer = self.load_top_layer()
        # Initialize Pi camera
        self.camera = PiCamera(resolution=(480, 480), framerate=15) 
        print("Initialize camera sensor...")
        time.sleep(1)

    # LOAD MOBILENET pre-trained model
    def load_mobile_net(self, resize):
        print("Loading mobile net...")
        shape = (IMAGE_SIZE, IMAGE_SIZE, 3) # input has form (height, width, channels)
        model = MobileNet(input_shape=shape,
                alpha=resize,
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                pooling=max)
        
        return model

    # Load FCL which will take in bottleneck features from mobilenet
    def load_top_layer(self):
        print("Loading top layer ...")
        return models.load_model(save_path)

    # Get all classfication labels
    def get_available_classes(self):
        return self.class_dictionary

    # get predicted values for a set of data
    def get_predictions(self, array, **args):
        if args.get("data_type") == "from_camera":
            bottleneck_data = self.restoredModel.predict(array)
            predictions_array = self.restoredTopLayer.predict(bottleneck_data)
            
        else:
            return args.get("data_type") +  " is Invalid data_type for function"

        # predicted class for each image                        
        predicted_labels = predictions_array.argmax(axis=-1) 

        for i, prediction in zip(range(len(predicted_labels)), predicted_labels):
            result = (self.class_dictionary.get(prediction), max(predictions_array[i]))
            # display the predictions on screen
            if args.get("show_all") == False:
	        return result # class with highest confidence
	    
	    elif args.get("show_all") == True:
                return predictions_array # confidence values for all classes

    # Get image data from camera as a numpy array with shape=(height,width,depth) 
    def get_camera_image_array(self):
        with PiRGBArray(self.camera, size=(IMAGE_SIZE, IMAGE_SIZE)) as output:            
            self.camera.capture(output, 'rgb', resize=(IMAGE_SIZE, IMAGE_SIZE))
            tensor = output.array*1. / 255 # Normalize rgb values
            tensor = np.array([tensor.astype('float32')], dtype='float32')
            output.truncate(0)
            return tensor

def main():
    cnn = PlantDetection()
    while True:    
        image_data = cnn.get_camera_image_array()
        prediction = cnn.get_predictions(image_data, data_type="from_camera", show_all=False)
        print prediction
        time.sleep(0.5)

    print prediction

if __name__ == "__main__":
    main()
