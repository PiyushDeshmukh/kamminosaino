#from neuralnets import *
from prepdata import *
import numpy as np
import tensorflow as tf
import random

import tflearn

input_nodes = 230400
hidden_nodes = 10 # change hidden nodes, this is only for development purpose
output_nodes = 25
learning_rate = 0.001

# Define the neural network
def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()

    # Inputs
    net = tflearn.input_data([None, trainX.shape[1]])

    # Hidden layer(s)
    net = tflearn.fully_connected(net, 1000, activation='tanh')
    #net = tflearn.fully_connected(net, 500, activation='tanh')

    # Output layer and training model
    net = tflearn.fully_connected(net, 25, activation='linear')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.001, loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    return model


if __name__ == '__main__':

    #print('Initializing Neural Net')
    #neural_net = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    #print('Created Neural Net with random weights')

    print('Fetching pixels...')
    trainX = []
    trainY = []
    gestures = fetch_pixels()
    random.shuffle(gestures)
    print("Done fetching pixels")

    print("Loading the inputs and outputs")
    for gesture in gestures:
        #inputs = values[0]
        #output = values[1]
        #outputs = [0] * output_nodes
        #outputs[ord(output) - ord('A')] += 1
        #gesture.set_output(outputs)
        trainX.append(gesture.get_pixels())
        trainY.append(gesture.get_outputs())
        #trainY.append(ord(output) - ord('A'))
        #neural_net.train(inputs, outputs)
    print('Done loaing!!')
    print(len(trainX), len(trainY))

    n = int(0.9 * len(trainX))

    # testX = []
    # for i in range(0, 240, 10):
    #     testX.append(trainX[i])
    # testX = np.array(testX)
    # for i in range(0, 240, 10):
    #     del trainX[i]
    # trainX = np.array(trainX)
    #
    #
    # testY = []
    # for i in range(0, 240, 10):
    #     testY.append(trainY[i])
    # testY = np.array(testY)
    # for i in range(0, 240, 10):
    #     del trainY[i]
    # trainY = np.array(trainY)

    testX = np.array(trainX[n:])
    trainX = np.array(trainX[:n])
    testY = np.array(trainY[n:])
    trainY = np.array(trainY[:n])

    print('TrainX : ', len(trainX))
    print('TrainY : ', len(trainY))
    print('TestX : ', len(testX))
    print('TestY : ', len(testY))

    print('Building model...')
    model = build_model()
    model.fit(trainX, trainY, validation_set=0.2, show_metric=True, batch_size=32, n_epoch=20)

    predictions = np.array(model.predict(testX)).argmax(axis=1)
    actual = testY.argmax(axis=1)
    for index, val in enumerate(predictions):
        print(val, actual[index])
    test_accuracy = np.mean(predictions == actual, axis=0)

    # Print out the result
    print("Test accuracy: ", test_accuracy)
