from network import *

print("\nLoading Iris training data ")
trainDataPath = "irisTrainData.txt"
trainDataset = IrisDataset(trainDataPath)

print("\nLoading Iris test data ")
testDataPath = "irisTestData.txt"
testDataset = IrisDataset(testDataPath)

net = BasicNeuralNetwork(mini_batch_size=8, layer_sizes=[100])
net.train(trainDataset, testDataset)
