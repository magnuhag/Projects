from time import perf_counter

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from NeuralNetwork import NeuralNet

# one-hot encoding the targest
def to_categorical_numpy(integerVector):
    nInputs = len(integerVector)
    nCategories = np.max(integerVector) + 1
    onehotVector = np.zeros((nInputs, nCategories))
    onehotVector[range(nInputs), integerVector] = 1
    return onehotVector

def fix_data():
    # download MNIST dataset
    digits = datasets.load_digits()
    # define inputs and labels
    inputs = digits.images
    labels = digits.target
    nInputs = len(inputs)
    inputs = inputs.reshape(nInputs, -1)
    X = inputs
    Y = to_categorical_numpy(labels)
    return X, Y

def main():

    X, Y = fix_data()
    XTrain, XTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2)

    inSize = len(X[0])
    out_size = len(yTest[0])

    Net = NeuralNet()
    #We don't need anything fancy for this demonstration
    Net.add(inSize, "sigmoid", inputSize = inSize)
    Net.add(10, "softmax")

    tTrainStart = perf_counter()

    #We're leaving batch size at a modest 10 as to not having to spend all day training the network
    Net.train(XTrain, yTrain, 100, "categorical_cross", "accuracy", 
              batchSize = 10, numIters = 100)

    tTrainStop = perf_counter()
    execTime = tTrainStop-tTrainStart
    print(f"The training took {execTime} seconds")

    pred = Net.predict(XTest)
    s = 0
    for i in range(len(XTest)):
        true = np.argmax(yTest[i])
        guess = np.argmax(pred[i])
        if true == guess:
            s += 1
    testAcc = s/len(yTest)
    print(f"test accuracy = {testAcc}")


if __name__ == "__main__":
    main()