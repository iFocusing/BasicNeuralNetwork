from network import *

print("\nLoading Iris training data ")
trainDataPath = "irisTrainData.txt"
trainDataset = IrisDataset(trainDataPath)

print("\nLoading Iris test data ")
testDataPath = "irisTestData.txt"
testDataset = IrisDataset(testDataPath)

net = BasicNeuralNetwork()
net.mbs = 0
net.train(trainDataset, testDataset)
