from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Sequential, models, utils
from data import cifar10
import numpy
import os

# all new operations will be in test mode from now on
K.set_learning_phase(0)  

# path to load the model that will be restored
saved_path = os.path.join(os.getcwd(), 'savedh5/cnn_model.h5')

# Load Test Data
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data() 
print('x_test shape:', test_features.shape)
print('testing samples', test_features.shape[0])

classes_tot = numpy.unique(train_labels).shape[0] # total number of classifications
print('classifications ', classes_tot)

# Normalize Test data
test_features = test_features.astype('float32')
test_features /= 255.0 # 255 is max colour intensity

# Convert labels to One-hot encoding 
test_labels = utils.to_categorical(test_labels, classes_tot)

restoredModel = models.load_model(saved_path) # load saved model
print("Loaded model from disk")
 
# evaluate loaded model on test data
score = restoredModel.evaluate(test_features, test_labels, verbose=0)
print("%s: %.2f%%" % (restoredModel.metrics_names[1], score[1]*100))
