from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Sequential, models, utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

IMAGE_SIZE = 150 # Dimensions of loaded image
BATCH_SIZE = 5
main_path = 'C:\\Users\\jjmiy_000\\Documents\\Github\\' # root directory
# path to load the model that will be restored
saved_direc = os.path.join(os.getcwd(), 'savedh5\cnn_model.h5')
# path to directory containing images to evaluate
eval_dataset = os.path.join(main_path, 'eval_dataset')

# all new operations will be in test mode from now on
K.set_learning_phase(0)

class PlantDetection:

    def __init__(self):
        # Configuration: Rescale RGB values of testing data
        self.test_gen = ImageDataGenerator(rescale=1.0 / 255,
                                    data_format='channels_last')
        # get dictionary of classes and invert dictionary to {0:"label_name", ...}
        self.class_dictionary = {0:"Broccoli", 1:"Cabbage", 2:"Onion", 3:"Tomato"}
        # load saved model
        self.restoredModel = models.load_model(saved_direc)
        print("Loaded model from disk")

    def get_available_classes(self):
            return self.class_dictionary

    # Prepare data as suitable input for Model
    def prepare_images(self, directory, mode):
        self.test_generator = self.test_gen.flow_from_directory(
                directory,
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                batch_size=BATCH_SIZE,
                shuffle=False,
                class_mode=mode)
        return self.test_generator

    # evaluate loaded model on prepared data from generator
    def test_model(self, generator, display):
        evaluations = self.restoredModel.evaluate_generator(generator)
        if display == True:
            print("loss: %5f, accuracy: %5f" % (evaluations[0], evaluations[1]))
        return evaluations

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
        return list_of_results

        def get_available_classes(self):
            return self.class_dictionary

if __name__ == "__main__":
    main()
