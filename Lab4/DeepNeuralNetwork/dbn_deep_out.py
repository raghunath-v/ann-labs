from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

import numpy as np
import os, matplotlib
from matplotlib import pyplot as plt

SEED = 1337
np.random.seed(SEED)

DATA_ROOT = 'data'

def load_file(file):
    with open(os.path.join(DATA_ROOT, file), 'r') as f:
        data = np.loadtxt(f, delimiter = ',', dtype = np.int)

    return data

def dbn_pipeline(data, epochs, encoding_dimensions = None):
    patterns = np.shape(data)[0]
    features = np.shape(data)[1]

    layers = []

    input_layer = BernoulliRBM(n_iter = epochs, n_components = encoding_dimensions[0], verbose = 1)
    layers.append(("input_layer", input_layer))

    for i, dim in enumerate(encoding_dimensions[1:]):
        hidden_layer = BernoulliRBM(n_iter = epochs, n_components = dim, verbose = 1)
        layers.append(("hidden_layer_{}".format(i), hidden_layer))

    dbn_pipeline = Pipeline(layers)

    return dbn_pipeline

def main():
    train_file = 'bindigit_trn.csv'
    train_target_file = 'targetdigit_trn.csv'

    test_file = 'bindigit_tst.csv'
    test_targets_file = 'targetdigit_tst.csv'

    data = load_file(train_file)
    targets = load_file(train_target_file)

    unique, counts = np.unique(targets, return_counts = True)
    print(dict(zip(unique, counts)))

    test_data = load_file(test_file)
    test_targets = load_file(test_targets_file)

    patterns = np.shape(data)[0]
    features = np.shape(data)[1]

    print(np.shape(data))
    print(np.shape(targets))

    CLASSES = len(unique)
    RBM_EPOCHS = 10
    MLP_EPOCHS = 10

    dbn = dbn_pipeline(data, RBM_EPOCHS, [150])
    dbn_output = dbn.fit_transform(data)

    # Define the deep model for the output
    model = Sequential()
    model.add(Dense(10, input_dim = np.shape(dbn_output)[1]))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(optimizer = "SGD", loss = "categorical_crossentropy", metrics = ["categorical_accuracy"])

    # Convert labels to categorical one-hot encoding
    one_hot_labels = keras.utils.to_categorical(targets, num_classes=CLASSES)
    
    model.fit(dbn_output, one_hot_labels, epochs = MLP_EPOCHS)

    # Try on test data
    test_dbn = dbn.transform(test_data)

    test_one_hot_labels = keras.utils.to_categorical(test_targets, num_classes=CLASSES)
    score = model.evaluate(test_dbn, test_one_hot_labels)

    print("Score: ", score)

if __name__ == "__main__":
    main()