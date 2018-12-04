import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from scipy.stats import truncnorm


class Dataset:
    def __init__(self):
        self.index = 0          # TODO: what this index is for? For now I use it for iteration index during the training

        self.obs = []           # for irisData is like [array(5.0,3.5,1.3,0.3), array(5.0,3.5,1.3,0.3), ... , ]
        self.classes = []       # for irisData is [array(1,0,0), array(0,1,0), array(0,0,1), ... , ]
        self.num_obs = 0        # number of observations
        self.num_classes = 0    # number of classes, for irisData is 3
        self.indices = []       # [0, 1, 2, 3, 4, ... , num_obs]

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_obs:      # index is greater than num_obs, then interrupt the iteration(read file iteration?)
            self.index = 0
            raise StopIteration
        else:
            self.index += 1
            return self.obs[self.index - 1], self.classes[self.index - 1]

    def reset(self):
        self.index = 0

    def get_obs_with_target(self, k):
        """
        from self.classes read the index_list, which the class is a particular k like (1,0,0)
        :param k:
        :return: list containing all the observations [array(5.0,3.5,1.3,0.3), array(4.9,3.0,1.4,0.2), ...]
                    with label k(1,0,0)
        """
        index_list = [index for index, value in enumerate(self.classes) if value == k] #
        return [self.obs[i] for i in index_list]

    def get_all_obs_class(self, shuffle=False):
        """
        shuffle indices, get all the pairs in that order.
        :param shuffle:
        :return: list containing all the observations with target [(array(5.0,3.5,1.3,0.3),array(1,0,0)),
                                                                    (array(4.9,3.0,1.4,0.2),array(1,0,0)), ... ]
        """
        if shuffle:
            random.shuffle(self.indices)
        return [(self.obs[i], self.classes[i]) for i in self.indices]

    def get_mini_batches(self, batch_size, shuffle=False):
        if shuffle:
            random.shuffle(self.indices)

        batches = [(self.obs[self.indices[n:n + batch_size]],
                    self.classes[self.indices[n:n + batch_size]])
                   for n in range(0, self.num_obs, batch_size)]
        return batches


class IrisDataset(Dataset):
    def __init__(self, path):
        super(IrisDataset, self).__init__()
        self.file_path = path
        self.loadFile()
        self.indices = np.arange(self.num_obs)

    def loadFile(self):
        # load a comma-delimited text file into an np matrix
        resultList = []
        f = open(self.file_path, 'r')
        for line in f:
            line = line.rstrip('\n')  # "1.0,2.0,3.0"
            sVals = line.split(',')  # ["1.0", "2.0, "3.0"]
            fVals = list(map(np.float32, sVals))  # [1.0, 2.0, 3.0]
            resultList.append(fVals)  # [[1.0, 2.0, 3.0] , [4.0, 5.0, 6.0]]
        f.close()
        data = np.asarray(resultList, dtype=np.float32)  # not necessary
        self.obs = data[:, 0:4]
        self.classes = data[:, 4:7]
        self.num_obs = data.shape[0]
        self.num_classes = 3


# Activations
def tanh(x, deriv=False):
    """
    d/dx tanh(x) = 1 - tanh^2(x)
    during backpropagation when we need to go though the derivative we have already computed tanh(x),
    therefore we pass tanh(x) to the function which reduces the gradient to:
    1 - tanh(x)
    """
    if deriv:
        return 1.0 - np.tanh(x)
    else:
        return np.tanh(x)


def sigmoid(x, deriv=False):
    """
    Task 2a
    This function is the sigmoid function. It gets an input digit or vector and should return sigmoid(x).
    The parameter "deriv" toggles between the sigmoid and the derivate of the sigmoid. Hint: In the case of the derivate
    you can expect the input to be sigmoid(x) instead of x
    :param x:       type: np.array
    :param deriv:   type: Boolean
    :return:        type: np.array
    """
    if deriv:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


def softmax(x, deriv=False):
    """
    Task 2a
    This function is the sigmoid function with a softmax applied. This will be used in the last layer of the network
    The derivate will be the same as of sigmoid(x)
    :param x:       type: np.array
    :param deriv:   type: Boolean
    :return:        type: np.array
    """
    if deriv:
        return x * (1 - x)
    else:
        exps = np.exp(x)
        return exps / np.sum(exps)


class Layer:
    def __init__(self, numInput, numOutput, activation=sigmoid):
        # print('Create layer with: {}x{} @ {}'.format(numInput, numOutput, activation))
        self.ni = numInput
        self.no = numOutput
        self.weights = np.zeros(shape=[self.ni, self.no], dtype=np.float32)
        self.biases = np.zeros(shape=[self.no], dtype=np.float32)
        self.initializeWeights()

        self.activation = activation
        self.last_input = None  # placeholder, can be used in backpropagation  -- I use this to store the input of this layer: y_{l-1}
        self.last_output = None  # placeholder, can be used in backpropagation -- output of inference function: y_{l}
        self.last_nodes = None  # placeholder, can be used in backpropagation  -- TODO: the value of nodes: z_{l}, but I don't use it

    def initializeWeights(self):
        """
        Task 2d
        Initialized the weight matrix of the layer. Weights should be initialized to something other than 0.
        You can search the literature for possible initialization methods.
        :return: None
        """
        # self.weights = np.random.randn(self.ni, self.no) * np.sqrt(2 / self.ni)
        self.weights = np.random.uniform(-0.1, 0.1, size=(self.ni, self.no))
        self.biases = np.random.uniform(-0.1, 0.1, size=(self.no))

    def inference(self, x):
        """
        Task 2b
        This transforms the input x with the layers weights and bias and applies the activation function
        Hint: you should save the input and output of this function usage in the backpropagation
        :param x:
        :return: output of the layer
        :rtype: np.array
        """
        self.last_input = x
        self.last_nodes = np.matmul(x.reshape(1, self.ni), self.weights) + self.biases
        self.last_output = self.activation(self.last_nodes)
        return self.last_output

    def backprop(self, error):
        """
        Task 2c
        This function applied the backpropagation of the error signal. The Layer receives the error signal from the following
        layer or the network. You need to calculate the error signal for the next layer by backpropagating thru this layer.
         You also need to compute the gradients for the weights and bias.
        :param error:
        :return: error signal for the preceeding layer
        :return: gradients for the weight matrix
        :return: gradients for the bias
        :rtype: np.array
        """
        # print('-------------Begin: backprop-----------')
        # print('Before backprop, weithes: \n', self.weights, '\n', 'biases: \n', self.biases)
        gradients_weight = np.matmul(self.last_input.reshape(self.ni, 1), \
                                     (error * sigmoid(self.last_output, True)).reshape(1, self.no))
        gradients_bias = error * sigmoid(self.last_output, True)
        # print("gradients_weight: ", gradients_weight, '\n', 'gradients_bias:', gradients_bias, '\n')
        error_signal = np.matmul((error * sigmoid(self.last_output, True)).reshape(1, self.no), self.weights.T)
        # print("error_signal: ", error_signal, "in layer:", self.ni, self.no)
        # print('After backprop, weithes: \n', self.weights, '\n', 'biases: \n', self.biases)
        # print('-------------End: have done backprop once-----------')
        return gradients_weight, gradients_bias, error_signal


# we apply softmax on the output layer, so the error of particular x_n is the log(output).

class BasicNeuralNetwork():
    def __init__(self, layer_sizes=[5], num_input=4, num_output=3, num_epoch=50, learning_rate=0.1,
                 mini_batch_size=8):
        self.layers = []       # to store the Object layer. [layers[0](form input to hidden layer 1),
                                                            # layers[1], ... ,
                                                            # layers[len(self.ls)]] (layers[3])
        self.ls = layer_sizes  # I consider this to be the size of different hidden layers [5,5,4] (len(self.ls) = 3),
        # don't contain the size of input and output layer.  The whole structure should be  4 + [5, 5, 4] + 3;
        self.ni = num_input
        self.no = num_output
        self.lr = learning_rate
        self.num_epoch = num_epoch
        self.mbs = mini_batch_size

        self.constructNetwork()

    def forward(self, x):
        """
        Task 2b
        This function forwards a single feature vector through every layer and return the output of the last layer
        :param x: input feature vector
        :return: output of the network
        :rtype: np.array
        """
        self.layers[0].inference(x)
        for i in range(1, len(self.ls)):
            self.layers[i].inference(self.layers[i - 1].last_output)
        output = self.layers[len(self.ls)].inference(self.layers[len(self.ls) - 1].last_output)
        # print("------", output, "-------")
        return output

    def train(self, train_dataset, eval_dataset=None, monitor_ce_train=True, monitor_accuracy_train=True,
              monitor_ce_eval=True, monitor_accuracy_eval=True, monitor_plot='monitor.png'):
        ce_train_array = []
        ce_eval_array = []
        acc_train_array = []
        acc_eval_array = []
        for e in range(self.num_epoch):
            if self.mbs:
                self.mini_batch_SGD(train_dataset)
            else:
                self.online_SGD(train_dataset)
            print('Finished training epoch: {}'.format(e))
            if monitor_ce_train:
                ce_train = self.ce(train_dataset)
                ce_train_array.append(ce_train)
                print('CE (train): {}'.format(ce_train))
            if monitor_accuracy_train:
                acc_train = self.accuracy(train_dataset)
                acc_train_array.append(acc_train)
                print('Accuracy (train): {}'.format(acc_train))
            if monitor_ce_eval:
                ce_eval = self.ce(eval_dataset)
                ce_eval_array.append(ce_eval)
                print('CE (eval): {}'.format(ce_eval))
            if monitor_accuracy_eval:
                acc_eval = self.accuracy(eval_dataset)
                acc_eval_array.append(acc_eval)
                print('Accuracy (eval): {}'.format(acc_eval))

        if monitor_plot:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
            line1, = ax[0].plot(ce_train_array, '--', linewidth=2, label='ce_train')
            line2, = ax[0].plot(ce_eval_array, label='ce_eval')

            line3, = ax[1].plot(acc_train_array, '--', linewidth=2, label='acc_train')
            line4, = ax[1].plot(acc_eval_array, label='acc_eval')

            ax[0].legend(loc='upper right')
            ax[1].legend(loc='upper left')
            ax[1].set_ylim([0, 1])

            plt.savefig(monitor_plot)

    def online_SGD(self, dataset):
        """
        Task 2d
        This function trains the network in an online fashion. Meaning the weights are updated after each observation.
        :param dataset:
        :return: None
        """
        dataset = dataset.get_all_obs_class(True)
        for item in dataset:
            output = self.forward(item[0])
            # error_signal = - item[1] * (1 - output)
            # error_signal = 1 - output
            error_signal = output - item[1]
            for i in range(len(self.ls), -1, -1):
                gradients_weight, gradients_bias, error_signal = self.layers[i].backprop(error_signal)
                self.layers[i].weights = self.layers[i].weights - self.lr * gradients_weight
                self.layers[i].biases = self.layers[i].biases - self.lr * gradients_bias


    def mini_batch_SGD(self, dataset):
        """
        Task 2d
        This function trains the network using mini batches. Meaning the weights updates are accumulated and applied after each mini batch.
        :param dataset:
        :return: None
        """
        dataset.reset()
        mini_batches = dataset.get_mini_batches(self.mbs)
        for item in mini_batches:
            error_signal = 0
            for i in range(self.mbs):
                output = self.forward(item[0][i])
                error_signal += output - item[1][i]
            error_signal = error_signal / self.mbs
            for k in range(len(self.ls), -1, -1):
                gradients_weight, gradients_bias, error_signal = self.layers[k].backprop(error_signal)
                self.layers[k].weights = self.layers[k].weights - self.lr * gradients_weight
                self.layers[k].biases = self.layers[k].biases - self.lr * gradients_bias

    def constructNetwork(self):
        """
        Task 2d
        uses self.ls self.ni and self.no to construct a list of layers. The last layer should use sigmoid_softmax as an activation function. any preceeding layers should use sigmoid.
        :return: None
        """
        l = Layer(self.ni, self.ls[0], sigmoid)
        l.initializeWeights()
        self.layers.append(l)
        for i in range(1, len(self.ls)):
            l = Layer(self.ls[i - 1], self.ls[i], sigmoid)
            l.initializeWeights()
            self.layers.append(l)
        l = Layer(self.ls[len(self.ls) - 1], self.no, softmax)
        l.initializeWeights()
        self.layers.append(l)

    def ce(self, dataset):
        ce = 0
        for x, t in dataset:
            t_hat = self.forward(x)
            ce += np.sum(np.nan_to_num(-t * np.log(t_hat) - (1 - t) * np.log(1 - t_hat)))
        return ce / dataset.num_obs

    def accuracy(self, dataset):
        cm = np.zeros(shape=[dataset.num_classes, dataset.num_classes], dtype=np.int)
        for x, t in dataset:
            t_hat = self.forward(x)
            c_hat = np.argmax(t_hat)  # index of largest output value
            c = np.argmax(t)
            cm[c, c_hat] += 1

        correct = np.trace(cm)
        return correct / dataset.num_obs

    def load(self, path=None):
        if not path:
            path = './network.save'
        with open(path, 'rb') as f:
            self.layers = pickle.load(f)

    def save(self, path=None):
        if not path:
            path = './network.save'
        with open(path, 'wb') as f:
            pickle.dump(self.layers, f)
