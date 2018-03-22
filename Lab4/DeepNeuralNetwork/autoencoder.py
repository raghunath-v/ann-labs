import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.optimizers import SGD
from keras.utils import plot_model


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


def autoencoder(nb_hidden_layers, X_train, batch_size, epochs):
    patterns = np.shape(X_train)[0]
    features = np.shape(X_train)[1]

    trained_encoders = []
    trained_decoders = []
    trained_autoencoders = []
    X_train_tmp = X_train

    # nb_hidden_layers = [features, 150, 100, 75]

    # Create an autoencoder for each pair of layers
    for n_in, n_out in zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]):
        print("Defining autoencoder for: {} -> {} -> {}".format(n_in, n_out, n_in))
        input = Input(shape = (n_in, ))
        encoding_layer = Dense(n_out, activation = "sigmoid")(input)
        
        decoding_layer = Dense(n_in, activation = "sigmoid")(encoding_layer)

        autoencoder = Model(inputs = input, outputs = decoding_layer)

        autoencoder.compile(loss='mse', optimizer='SGD')

        plot_model(autoencoder, show_shapes = True, to_file='autoencoder_{}.png'.format(n_in))

        if len(trained_autoencoders) != 0 :
            X_train_tmp = trained_autoencoders[-1].get_output_at(0)
            print("shape X_train_tmp: ", np.shape(X_train_tmp))

        autoencoder.fit(X_train_tmp, X_train_tmp, batch_size = batch_size, epochs = epochs)

        trained_autoencoders.append(autoencoder)
        trained_encoders.append(autoencoder.layers[0])
        trained_decoders.append(autoencoder.layers[1])

        X_train_tmp = autoencoder.predict(X_train_tmp)
        

    print("trained encoders: {}".format(np.shape(trained_encoders)))
    print("trained decoders: {}".format(np.shape(trained_decoders)))

    return trained_encoders, trained_decoders

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

    CLASSES = len(unique)
    MLP_EPOCHS = 10
    BATCHSIZE = 32

    nb_hidden_layers = [features, 150, 100, 75]

    trained_encoders, trained_decoders = autoencoder(nb_hidden_layers, data, BATCHSIZE, MLP_EPOCHS)

    model = Sequential()
    for encoding_layer in trained_encoders:
        model.add(encoding_layer)

    for decoding_layer in trained_decoders:
        model.add(decoding_layer)
    
    model.add(Dense(10, input_dim = nb_hidden_layers[0]))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(optimizer = "SGD", loss = "categorical_crossentropy", metrics = ["categorical_accuracy"])
    
    plot_model(model, to_file='autoencoder.png')
    # Convert labels to categorical one-hot encoding
    one_hot_labels = keras.utils.to_categorical(targets, num_classes=CLASSES)
    
    model.fit(data, one_hot_labels, epochs = MLP_EPOCHS)

    # Try on test data
    test_one_hot_labels = keras.utils.to_categorical(test_targets, num_classes=CLASSES)
    score = model.evaluate(test_data, test_one_hot_labels)

    print("Score: ", score)
if __name__ == "__main__":
    main()