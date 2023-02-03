from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Sequential, models, utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import os, time, CNN
import numpy as np

IMAGE_SIZE = 32 # Dimensions of loaded image
BATCH_SIZE = 20
main_path = 'C:\\Users\\jjmiy_000\\Documents\\Github\\' # root directory
# path to load the model that will be restored
saved_direc = os.path.join(os.getcwd(), 'savedh5\cnn_model_6.h5')
# path to directory containing images to evaluate
eval_dataset = os.path.join(main_path, 'eval_dataset_2')

# all new operations will be in test mode from now on
K.set_learning_phase(0)

def load_model_from_weights():
    model = CNN.new_model()
    model.load_weights(saved_direc)
    return model

class PlantDetection:

    def __init__(self):
        # Configuration: Rescale RGB values of testing data
        self.test_gen = ImageDataGenerator(rescale=1.0 / 255,
                                    data_format='channels_last')
        # get dictionary of classes and invert dictionary to {0:"label_name", ...}
        self.class_dictionary = {0:"Broccoli", 1:"Cabbage", 2:"Onion", 3:"Spinach", 4:"Strawberry", 5: "Tomato"}
        # load saved model
        self.restoredModel = models.load_model(saved_direc)
        print("Loaded model from disk")

    def get_available_classes(self):
            return self.class_dictionary

    # Prepare data as suitable input for Model
    def prepare_images(self, **args):
        try:
            # Get image data from a directory
            directory = args["directory"]
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
            return data # return numpy array

    # evaluate loaded model on prepared data from generator Note: Only works when
    #    input data [subdirectories] matches the number of classes == 4
    def test_model(self, generator, display):
        evaluations = self.restoredModel.evaluate_generator(generator)
        if display == True:
            print("loss: %5f, accuracy: %5f" % (evaluations[0], evaluations[1]))
        return evaluations

    # Get all classfication labels
    def get_available_classes(self):
        return self.class_dictionary

    # Get path to saved pictures from pi camera
    def get_data_path(self):
        return eval_dataset

    # get predicted values for a set of data
    def get_predictions(self, generator, display):
        predictions_array = self.restoredModel.predict_generator(generator) # numpy array of predicions
        predicted_labels = predictions_array.argmax(axis=-1) # predicted class/label for each image
        list_of_results = [] # Hold predicted labels with corresponding scores

        for i, prediction in zip(range(len(predicted_labels)), predicted_labels):
            result = (self.class_dictionary.get(prediction), max(predictions_array[i]))
            list_of_results.append(result)
            # display the predictions on screen
            if display == True:
                correct_labels = generator.classes # get verified labels of input test data
                print(" Correct label - %s, Predicted label - %s, Score: [%5f]"
                    % (self.class_dictionary.get(correct_labels[i]), result[0],
                        result[1]))
        return predictions_array

    # Get image data from camera to input to inference model
    def get_image_data(self):
##        with PiCamera as pi_cam:
##            pi_cam = PiCamera()    # initialize the raspi camera
##            pi_cam.resolution = (150, 150)
##            raw = PiRGBArray(pi_cam, size=(150, 150)) # Capture camera stream directly
##            time.sleep(0.2) # wait for camera sensor activation
##            pi_cam.capture(raw, format='rgb') # Captured image in rgb format
##            frame = raw.array # get image as numpy array
        return frame
def main():
    detector = PlantDetection()
    generator = detector.prepare_images(directory=eval_dataset,
                                        class_mode='categorical')
    array = detector.get_predictions(generator, display=True)
    #print("\n\n", array)
    detector.test_model(generator, display=True)
if __name__ == "__main__":
    main()
