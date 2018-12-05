from network import *

print("\nLoading Iris training data ")
trainDataPath = "irisTrainData.txt"
trainDataset = IrisDataset(trainDataPath)

print("\nLoading Iris test data ")
testDataPath = "irisTestData.txt"
testDataset = IrisDataset(testDataPath)

type = 'random'
# type = 'uniform'
# type = 'gaussian'
net = BasicNeuralNetwork(mini_batch_size=0, layer_sizes=[5], num_epoch=50, learning_rate=0.1, type_of_initial_weights=type)
# net = BasicNeuralNetwork(mini_batch_size=8, layer_sizes=[5], num_epoch=50, learning_rate=0.1, type_of_initial_weights=type)
#
# # tuning mini_batch_size
# net = BasicNeuralNetwork(mini_batch_size=16, layer_sizes=[5], num_epoch=50, learning_rate=0.1, type_of_initial_weights=type)
# net = BasicNeuralNetwork(mini_batch_size=32, layer_sizes=[5], num_epoch=50, learning_rate=0.1, type_of_initial_weights=type)
#
# # tuning the size of nodes in one layer
# net = BasicNeuralNetwork(mini_batch_size=8, layer_sizes=[50], num_epoch=50, learning_rate=0.1, type_of_initial_weights=type)
# net = BasicNeuralNetwork(mini_batch_size=8, layer_sizes=[500], num_epoch=50, learning_rate=0.1, type_of_initial_weights=type)
#
# # adding layers
# net = BasicNeuralNetwork(mini_batch_size=8, layer_sizes=[5,5], num_epoch=50, learning_rate=0.1, type_of_initial_weights=type)
# net = BasicNeuralNetwork(mini_batch_size=8, layer_sizes=[5,6,5], num_epoch=50, learning_rate=0.1, type_of_initial_weights=type)
# net = BasicNeuralNetwork(mini_batch_size=8, layer_sizes=[4,5,3], num_epoch=50, learning_rate=0.1, type_of_initial_weights=type)
#
# # tuning num_epoch
# net = BasicNeuralNetwork(mini_batch_size=8, layer_sizes=[5], num_epoch=100, learning_rate=0.1, type_of_initial_weights=type)
# net = BasicNeuralNetwork(mini_batch_size=8, layer_sizes=[5], num_epoch=150, learning_rate=0.1, type_of_initial_weights=type)
# net = BasicNeuralNetwork(mini_batch_size=8, layer_sizes=[5], num_epoch=200, learning_rate=0.1, type_of_initial_weights=type)
#
# # tuning learning_rate
# net = BasicNeuralNetwork(mini_batch_size=8, layer_sizes=[5], num_epoch=50, learning_rate=0.4, type_of_initial_weights=type)
# net = BasicNeuralNetwork(mini_batch_size=8, layer_sizes=[5], num_epoch=50, learning_rate=0.8, type_of_initial_weights=type)
#

net.train(trainDataset, testDataset)

