from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Sequential, models, utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

IMAGE_SIZE = 150 # Dimensions of loaded image
BATCH_SIZE = 24
main_path = 'C:\\Users\\jjmiy_000\\Documents\\REPOSITORY' # root directory
# path to load the model that will be restored
saved_model = os.path.join(main_path, 'savedh5\cnn_model.h5')
# path to model evaluating data
eval_dataset = os.path.join(main_path, 'eval_dataset')

# all new operations will be in test mode from now on
K.set_learning_phase(0)

class PlantDetection:

    def __init__(self):
        # Configuration: Rescale RGB values of testing data
        self.test_gen = ImageDataGenerator(rescale=1.0 / 255,
                                    data_format='channels_last')
        # load saved model
        self.restoredModel = models.load_model(saved_model)
        print("Loaded model from disk")

    # Prepare data as suitable input for Model
    def prepare_images(self, directory, total):
        self.test_generator = test_gen.flow_from_directory(
                directory,
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                batch_size=total,
                shuffle=False,
                class_mode='categorical')
        return self.test_generator

    # evaluate loaded model on prepared data from generator
    def test_model(self, generator, display):
        evaluations = restoredModel.evaluate_generator(generator)
        if display == 1:
            print("loss: %5f, accuracy: %5f" % (evaluations[0], evaluations[1]))
        return evaluations

    # get predicted values
    def get_predictions(self, generator, display):
        predictions_array = self.restoredModel.predict_generator(generator) # numpy array of predicions
        predicted_labels = predictions_array.argmax(axis=-1) # predicted class/label for each image
        # get dictionary of classes and invert dictionary to {0:"label_name", ...}
        self.class_dictionary = {v: k for k, v in generator.class_indices.items()}
        correct_labels = generator.classes # get verified labels of input test data

        if display == 1: # display the predictions on screen
            for i, prediction in zip(range(len(predicted_labels)), predicted_labels):
                print("prediction - %s | Actual label - %s | Score: [%5f]" %
                    (self.class_dictionary.get(prediction),
                     self.class_dictionary.get(correct_labels[i]),
                     max(predictions_array[i])))

        return predicted_labels, predictions_array

if __name__ == "__main__":
    main()
