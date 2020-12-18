from __future__ import division  # floating point division
import numpy as np
import utilities as utils
from scipy.spatial.distance import hamming

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it should ignore this last feature
        self.params = {'usecolumnones': True}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.means = []
        self.stds = []
        self.numfeatures = 0
        self.numclasses = 0

    def classprobabilities(self, Xtest): #calculates the probabilty of test data for all classes
        def normpdf(x, mean, std):
            return (1 / (np.sqrt(2*np.pi) * std + 1e-8)) * np.exp(-(np.square(x-mean)/(2*np.square(std) + 1e-8)))
        
        probabilities = np.ones((Xtest.shape[0], self.numclasses))
        for i in range(self.numclasses):
            for j in range(self.numfeatures):
                probabilities[:,i] = np.multiply(probabilities[:,i], normpdf(Xtest[:,j], self.means[i,j], self.stds[i,j]))
            probabilities[:,i] = np.multiply(probabilities[:,i], self.priors[i])
        return probabilities
            

    def learn(self, Xtrain, ytrain):
        
        if (self.params['usecolumnones'] == False):
            Xtrain = Xtrain[:, :-1]
        self.classes, class_counts = np.unique(ytrain, return_counts=True)
        self.priors = np.divide(class_counts, ytrain.shape[0])
        
        self.numclasses = len(self.classes)
        self.numfeatures = Xtrain.shape[1]

        origin_shape = (self.numclasses, self.numfeatures)
        self.means = np.zeros(origin_shape)
        self.stds = np.zeros(origin_shape)
        for i in range(len(self.classes)):
            self.means[i] = np.mean(Xtrain[ytrain == self.classes[i]], axis=0)
            self.stds[i] = np.std(Xtrain[ytrain == self.classes[i]], axis=0)

        assert self.means.shape == origin_shape
        assert self.stds.shape == origin_shape

    def predict(self, Xtest):

        ytest = np.zeros(Xtest.shape[0], dtype=int)
        if (self.params['usecolumnones'] == False):
            Xtest = Xtest[:, :-1]
        probabilities = NaiveBayes.classprobabilities(self, Xtest)
        output = np.argmax(probabilities, axis=1)
        ytest = [self.classes[x] for x in output]    # if labels are not 0 & 1

        assert len(ytest) == Xtest.shape[0]
        return ytest

class LogitReg(Classifier):

    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.00, 'lr': 0.01, 'epochs': 1000, 'batch_size': 32, 'regularizer': 'none'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """
        cost = 0.0
        y_hat = utils.sigmoid(np.dot(X, theta))
        cost = (np.dot(-y.T, np.log(y_hat)) - np.dot((1 - y).T, np.log(1 - y_hat)))/X.shape[0] + self.params['regwgt']*self.regularizer[0](theta)

        return cost

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """
        grad = np.zeros(len(theta))
        y_hat = utils.sigmoid(np.dot(X, theta))
        grad = np.dot(X.T, (y_hat - y)) / y.shape[0] + self.params['regwgt']*self.regularizer[1](theta)

        return grad

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """
        self.weights = np.zeros(Xtrain.shape[1])
        numsamples = Xtrain.shape[0]
        for i in range(self.params['epochs']):
            permutation = list(np.random.permutation(numsamples))
            shuffled_X = Xtrain[permutation, :]
            shuffled_y = ytrain[permutation]
            for j in range(0, numsamples, self.params['batch_size']):
                mini_X = shuffled_X[j:j + self.params['batch_size'], :]
                mini_y = shuffled_y[j:j + self.params['batch_size']]
                g = LogitReg.logit_cost_grad(self, self.weights, mini_X, mini_y)
                self.weights = np.subtract(self.weights, np.multiply(self.params['lr'], g)) # w = w - lr * grad
        print('logit cost:', self.logit_cost(self.weights, Xtrain, ytrain))

    def predict(self, Xtest):

        ytest = np.zeros(Xtest.shape[0], dtype=int)
        ytest = utils.sigmoid(np.dot(Xtest, self.weights)) >= 0.5

        assert len(ytest) == Xtest.shape[0]
        return ytest

class NeuralNet(Classifier):
    """
    a neural network with a single hidden layer. Cross entropy is used as the cost function.

    Parameters:
    nh -- number of hidden units
    transfer -- transfer function, in this case, sigmoid
    stepsize -- stepsize for gradient descent
    epochs -- learning epochs
    """
    def __init__(self, parameters={}):
        self.params = {'nh': 16,
                    'transfer': 'sigmoid',
                    'stepsize': 0.01,
                    'batch_size': 32,
                    'epochs': 100}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.w_input = None
        self.w_output = None

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        # hidden activations
        a_hidden = self.transfer(np.dot(self.w_input, inputs))

        # output activations
        a_output = self.transfer(np.dot(self.w_output, a_hidden))

        return (a_hidden, a_output)

    def backprop(self, x, y):
        """
        Returns a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """

        d1 = np.subtract(self.a[1], y)
        nabla_output = np.dot(d1, self.a[0].T)
        d2 = np.multiply(np.dot(self.w_output.T, d1), np.multiply(self.a[0], np.subtract(1, self.a[0])))
        nabla_input = np.dot(d2, x.T)

        assert nabla_input.shape == self.w_input.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_input, nabla_output)

    def learn(self, Xtrain, ytrain):
        n_x = Xtrain.shape[1]
        n_y = 1 #ytrain.shape[1]
        self.w_input = np.random.randn(self.params['nh'], n_x)
        self.w_output = np.random.randn(n_y, self.params['nh'])
        numsamples = Xtrain.shape[0]
        for i in range(self.params['epochs']):
            permutation = list(np.random.permutation(numsamples))
            shuffled_X = Xtrain[permutation, :]
            shuffled_y = ytrain[permutation]
            for j in range(0, numsamples, self.params['batch_size']):
                mini_X = shuffled_X[j:j + self.params['batch_size'], :]
                mini_y = shuffled_y[j:j + self.params['batch_size']]
                inputs = mini_X.T
                self.a = self.feedforward(inputs)
                (nabla_input, nabla_output) = self.backprop(inputs, mini_y)
                self.w_input = np.subtract(self.w_input, np.multiply(self.params['stepsize'], nabla_input))
                self.w_output = np.subtract(self.w_output, np.multiply(self.params['stepsize'], nabla_output))

    def predict(self, Xtest):
        ytest = np.zeros(Xtest.shape[0], dtype=int)
        inputs = Xtest.T
        a = self.feedforward(inputs)
        ytest = a[1].T >= 0.5

        assert len(ytest) == Xtest.shape[0]
        return ytest

class NeuralNet2(Classifier):
    """ Neural network with 2 hidden layers """

    def __init__(self, parameters={}):
        self.params = {'nh': 16,
                    'transfer': 'sigmoid',
                    'stepsize': 0.01,
                    'decay': 0.9,
                    'batch_size': 32,
                    'epochs': 100}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.w_0 = None
        self.w_1 = None
        self.w_output = None

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        # hidden activations
        a_0 = self.transfer(np.dot(self.w_0, inputs))
        a_1 = self.transfer(np.dot(self.w_1, a_0))
        # output activations
        a_output = self.transfer(np.dot(self.w_output, a_1))

        return (a_0, a_1, a_output)

    def backprop(self, x, y):
        """
        Returns a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """

        d1 = np.subtract(self.a[2], y)
        nabla_output = np.dot(d1, self.a[1].T)
        d2 = np.multiply(np.dot(self.w_output.T, d1), np.multiply(self.a[1], np.subtract(1, self.a[1])))
        nabla_1 = np.dot(d2, self.a[0].T)
        d3 = np.multiply(np.dot(self.w_1.T, d2), np.multiply(self.a[0], np.subtract(1, self.a[0])))
        nabla_0 = np.dot(d3, x.T)

        assert nabla_0.shape == self.w_0.shape
        assert nabla_1.shape == self.w_1.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_0, nabla_1, nabla_output)

    def learn(self, Xtrain, ytrain):
        n_x = Xtrain.shape[1]
        n_y = 1  # ytrain.shape[1]
        self.w_0 = np.random.randn(self.params['nh'], n_x)
        self.w_1 = np.random.randn(self.params['nh'], self.params['nh'])
        self.w_output = np.random.randn(n_y, self.params['nh'])
        numsamples = Xtrain.shape[0]
        s_0 = np.zeros_like(self.w_0)  # moving average of the squared gradient
        s_1 = np.zeros_like(self.w_1)
        s_2 = np.zeros_like(self.w_output)
        epsilon = 1e-8  # epsilon to be added to the denominator to avoid dividing by zero
        for i in range(self.params['epochs']):
            permutation = list(np.random.permutation(numsamples))
            shuffled_X = Xtrain[permutation, :]
            shuffled_y = ytrain[permutation]
            for j in range(0, numsamples, self.params['batch_size']):
                mini_X = shuffled_X[j:j + self.params['batch_size'], :]
                mini_y = shuffled_y[j:j + self.params['batch_size']]
                inputs = mini_X.T
                self.a = self.feedforward(inputs)
                g = self.backprop(inputs, mini_y)
                s_0 = np.add(np.dot(self.params['decay'], s_0), np.dot((1-self.params['decay']), np.square(g[0])))  # MeanSquare = decay*MeanSquare + (1-decay)* g^2
                s_1 = np.add(np.dot(self.params['decay'], s_1), np.dot((1-self.params['decay']), np.square(g[1])))
                s_2 = np.add(np.dot(self.params['decay'], s_2), np.dot((1-self.params['decay']), np.square(g[2])))
                self.w_0 = np.subtract(self.w_0, np.divide(np.dot(self.params['stepsize'], g[0]), np.add(np.sqrt(s_0), epsilon))) # w = w - ( step_size*g / (sqrt(MeanSquare) + eps) )
                self.w_1 = np.subtract(self.w_1, np.divide(np.dot(self.params['stepsize'], g[1]), np.add(np.sqrt(s_1), epsilon)))
                self.w_output = np.subtract(self.w_output, np.divide(np.dot(self.params['stepsize'], g[2]), np.add(np.sqrt(s_2), epsilon)))

    def predict(self, Xtest):

        ytest = np.zeros(Xtest.shape[0], dtype=int)

        inputs = Xtest.T
        a = self.feedforward(inputs)
        ytest = a[2].T >= 0.5

        assert len(ytest) == Xtest.shape[0]
        return ytest

class KernelLogitReg(LogitReg):
    """
    Similar to class LogitReg except one more parameter 'kernel': None, linear, or hamming)
    """
    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None', 'kernel': 'linear', 'k': 100}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):

        Ktrain = None  # Ktrain the is the kernel representation of the Xtrain

        permutation = list(np.random.permutation(Xtrain.shape[0]))
        shuffled_X = Xtrain[permutation]
        self.centers = shuffled_X[:self.params['k']]
        if (self.params['kernel'] == 'hamming'):
            Ktrain = np.zeros((Xtrain.shape[0], self.params['k']))
            for i in range(Xtrain.shape[0]):
                for j in range(self.params['k']):
                    Ktrain[i, j] = sum(f1 != f2 for f1, f2 in zip(list(Xtrain[i]), list(self.centers[j]))) 
        else:
            Ktrain = np.matmul(Xtrain, self.centers.T)

        self.weights = np.zeros(Ktrain.shape[1],)

        LogReg = LogitReg(parameters={'regwgt': self.params['regwgt'], 'regularizer': self.params['regularizer']})
        LogReg.learn(Ktrain, ytrain)
        self.weights = LogReg.weights

        self.transformed = Ktrain

    def predict(self, Xtest):
        ytest = np.zeros(Xtest.shape[0], dtype=int)
        if (self.params['kernel'] == 'hamming'):
            Ktest = np.zeros((Xtest.shape[0], self.params['k']))
            for i in range(Xtest.shape[0]):
                for j in range(self.params['k']):
                    Ktest[i, j] = sum(f1 != f2 for f1, f2 in zip(list(Xtest[i]), list(self.centers[j]))) 
        else:
            Ktest = np.matmul(Xtest, self.centers.T)

        ytest = utils.sigmoid(np.dot(Ktest, self.weights)) >= 0.5
        assert len(ytest) == Xtest.shape[0]
        return ytest


def test_lr():
    print("Basic test for logistic regression...")
    clf = LogitReg()
    theta = np.array([0.])
    X = np.array([[1.]])
    y = np.array([0])

    try:
        cost = clf.logit_cost(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost!")
    assert isinstance(cost, float), "logit_cost should return a float!"

    try:
        grad = clf.logit_cost_grad(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost_grad!")
    assert isinstance(grad, np.ndarray), "logit_cost_grad should return a numpy array!"

    print("Test passed!")
    print("-" * 50)

def test_nn():
    print("Basic test for neural network...")
    clf = NeuralNet()
    X = np.array([[1., 2.], [2., 1.]])
    y = np.array([0, 1])
    clf.learn(X, y)

    assert isinstance(clf.w_input, np.ndarray), "w_input should be a numpy array!"
    assert isinstance(clf.w_output, np.ndarray), "w_output should be a numpy array!"

    try:
        res = clf.feedforward(X[0, :])
    except:
        raise AssertionError("feedforward doesn't work!")

    try:
        res = clf.backprop(X[0, :], y[0])
    except:
        raise AssertionError("backprob doesn't work!")

    print("Test passed!")
    print("-" * 50)

def main():
    test_lr()
    test_nn()

if __name__ == "__main__":
    main()
