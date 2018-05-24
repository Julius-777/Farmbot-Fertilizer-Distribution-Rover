#!/usr/bin/python -W ignore::DeprecationWarning
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Sequential, models, utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from picamera.array import PiRGBArray
from picamera import PiCamera
import os, time, CNN
import numpy as np

IMAGE_SIZE = 32 # Dimensions of loaded image
BATCH_SIZE = 5
main_path = '/Home/pi/Kilimanjaro' # root directory
# path to load the model that will be restored
saved_direc = os.path.join(os.getcwd(), 'savedh5/cnn_model_weights_6.h5')
# path to directory containing images to evaluate
eval_dataset = os.path.join(main_path, 'eval_dataset')

# all new operations will be in test mode from now on
K.set_learning_phase(0)

def load_model_from_weights(directory):
    model = CNN.new_model()
    model.load_weights(directory)
    return model

class PlantDetection:

    def __init__(self):
        # Configuration: Rescale RGB values of testing data
        self.test_gen = ImageDataGenerator(rescale=1.0 / 255,
                                    data_format='channels_last')
        # get dictionary of classes and invert dictionary to {0:"label_name", ...}
        #self.class_dictionary = {0:"Broccoli", 1:"Cabbage", 2:"Onion", 3:"Tomato"}
        self.class_dictionary ={0:"Broccoli", 1:"Cabbage", 2:"Onion", 3:"Spinach", 4:"Strawberry", 5: "Tomato"}
        # load saved model
        self.restoredModel =  load_model_from_weights(saved_direc)# models.load_model(saved_direc)
        print("Loaded model from disk")
        self.camera = PiCamera(resolution=(720, 720), framerate=15) 
        print("Initialize camera sensor...")
        time.sleep(1)

    def get_available_classes(self):
            return self.class_dictionary

    # Prepare data as suitable input for Model
    def prepare_images(self, **args):
        try:
            # Get image data from a directory
            directory = args["from_directory"]
            mode = args["class_mode"]
            test_generator = self.test_gen.flow_from_directory(
                    directory,
                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    class_mode=mode)
            return test_generator
        # Get image data directly from pi camera
        except KeyError:
            data = None
            if args["from_camera"] == True:
                data = self.get_image_data()

        return data


    # evaluate loaded model on prepared data from generator Note: Only works when
    #    input data [subdirectories] matches the number of classes == 4
    def test_model(self, generator, display):
        evaluations = self.restoredModel.evaluate_generator(generator)
        if display == True:
            print "loss: %5f, accuracy: %5f" % (evaluations[0], evaluations[1])
        return evaluations

    # Get all classfication labels
    def get_available_classes(self):
        return self.class_dictionary

    # Get path to saved pictures from pi camera
    def get_data_path(self):
        return eval_dataset

    # get predicted values for a set of data
    def get_predictions(self, data, **args):
        if args.get("data_type") == "from_directory":
            # numpy array of predicions
            predictions_array = self.restoredModel.predict_generator(data)
        elif args.get("data_type") == "from_camera":
            predictions_array = self.restoredModel.predict(data)
        else:
            return args.get("data_type") +  " is Invalid data_type for function"
                                
        predicted_labels = predictions_array.argmax(axis=-1) # predicted class/label for each image
        list_of_results = '' # Hold predicted labels with corresponding scores

        for i, prediction in zip(range(len(predicted_labels)), predicted_labels):
            result = (self.class_dictionary.get(prediction), max(predictions_array[i]))
            # display the predictions on screen
            if args.get("display") == True and args.get("data_type")=="from_directory":
                correct_labels = generator.classes # get verified labels of input test data
                list_of_results += " Correct label - %s, Predicted label - %s,\
                Score: [%5f]\n" % (self.class_dictionary.get(correct_labels[i]), result[0], result[1])
            elif args.get("data_type") == "from_camera":
	        return result # prediction for single image
              
        return list_of_results # predicions made for set of data

    # Get image data from camera to input to inference model
    def get_image_data(self):
        with PiRGBArray(self.camera, size=(32, 32)) as output:            
            self.camera.capture(output, 'rgb', resize=(32, 32))
            tensor = output.array*1. / 255 # Normalize rgb values
            tensor = np.array([tensor.astype('float32')], dtype='float32')
            output.truncate(0)
            return tensor

def main():
    cnn = PlantDetection()
    direc = cnn.get_data_path()
    prediction = (0,0)
    while True:    
        data_generator = cnn.prepare_images(from_camera=True)
        prediction = cnn.get_predictions(data_generator, data_type="from_camera")
        print prediction
        time.sleep(0.5)

    print prediction

if __name__ == "__main__":
    main()
