from __future__ import division  # floating point division
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import utilities as utils

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.weights = None
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        self.weights = None
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
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.5}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)/numsamples + np.multiply(self.params['regwgt'], np.eye(Xtrain.shape[1]))), Xtrain.T),ytrain)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest

class SGD(Regressor):

    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'step_size': 0.01, 'epochs': 1000, 'Report_error_graph': False}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.random.rand(Xtrain.shape[1])
        if (self.params['Report_error_graph'] == True):
            times_graph = np.zeros(np.int(self.params['epochs']/25) +1) # for time calculation if 'Report_error_graph' is selected
            
            error_graph = np.zeros(np.int(self.params['epochs']/25))
            
        for i in range(self.params['epochs']):
            t0 = time.time()
            permutation = list(np.random.permutation(numsamples))
            shuffled_X = Xtrain[permutation, :]
            shuffled_y = ytrain[permutation].reshape((numsamples,1))
            for j in range(numsamples):
                g = np.dot(shuffled_X[j,:].reshape((Xtrain.shape[1],1)), np.subtract(np.dot(shuffled_X[j,:].T, self.weights), shuffled_y[j])) # ∇cj = (xj.T*w − yj)*xj
                self.weights = np.subtract(self.weights, np.dot(self.params['step_size'], g))
            t1 = time.time()
            if (self.params['Report_error_graph'] == True and i % 25 == 0):
                k = np.int(i/25)
                times_graph[k+1] = 1000*(t1-t0) + times_graph[k]
                error_graph[k] = np.square(np.linalg.norm(np.subtract(np.dot(Xtrain, self.weights),ytrain)))/ytrain.shape[0]
        if (self.params['Report_error_graph'] == True):
            plt.subplot(2, 1, 1)
            plt.plot(range(0, self.params['epochs'], 25), error_graph)
            plt.title('training error')
            plt.ylabel('error')
            plt.xlabel('epochs')
            plt.subplot(2, 1, 2)
            plt.plot(times_graph[1:], error_graph)
            plt.ylabel('error')
            plt.xlabel('time (ms)')
            plt.show()


    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest

class BatchGD(Regressor):

    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'step_size_max': 1.0, 'max_iterations': 1000}
        self.reset(parameters)

    def line_search(self, c, g, Xtrain, ytrain): # finds the suitable step size, and returns only the new weights
        numsamples = Xtrain.shape[0]
        step_size = self.params['step_size_max']
        t = 0.7
        tolerance = 10e-4
        obj = c
        max_iterations = 1000
        for i in range(max_iterations):
            w = np.subtract(self.weights, np.dot(step_size, g))
            c = np.square(np.linalg.norm(np.subtract(np.dot(Xtrain, w), ytrain)))/(2*numsamples) # c = ||x*w - y||^2 / 2*n
            if(c < obj - tolerance): # Improvement should be at least as much as tolerance
                break
            step_size = t*step_size
            if(i==max_iterations):
                return self.weights # Could not improve solution, return same weights
        return w

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.random.rand(Xtrain.shape[1])
        err = np.inf
        tolerance = 10e-4
        for i in range(self.params['max_iterations']):
            c = np.square(np.linalg.norm(np.subtract(np.dot(Xtrain, self.weights), ytrain)))/(2*numsamples) # c = ||x*w - y||^2 / 2*n
            if (np.abs(np.subtract(c, err)) < tolerance):
                break # The process only goes on while |c(w) − err| > tolerance and have not reached max iterations
            err = c
            g = np.dot(Xtrain.T, np.subtract(np.dot(Xtrain, self.weights), ytrain))/numsamples # ∇c = x.T*(x*w − y) / n
            self.weights = BatchGD.line_search(self, c, g, Xtrain, ytrain) # calling the line_search func defined above to get new weights

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest

class BatchGDLasso(Regressor):

    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'reg': 0.01, 'max_iterations': 1000}
        self.reset(parameters)

    def prox(self, step_size, w): #
        for i in range(len(w)):
            if (w[i] > step_size*self.params['reg']):
                w[i] = np.subtract(w[i], step_size*self.params['reg'])
            elif (w[i] < -step_size*self.params['reg']):
                w[i] = np.add(w[i], step_size*self.params['reg'])
            else:
                w[i] = 0
        return w

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.zeros(Xtrain.shape[1])
        err = np.inf
        tolerance = 10e-4
        XX = np.matmul(Xtrain.T, Xtrain)/numsamples
        Xy = np.matmul(Xtrain.T, ytrain)/numsamples
        step_size = 1/(2*(np.linalg.norm(XX, ord='fro')))
        for i in range(self.params['max_iterations']):
            c = np.square(np.linalg.norm(np.subtract(np.dot(Xtrain, self.weights), ytrain)))/(2*numsamples) # c = ||x*w - y||^2 / 2*n
            if (np.abs(np.subtract(c, err)) < tolerance):
                break # The process only goes on while |c(w) − err| > tolerance and have not reached max iterations
            err = c
            w = np.add(np.subtract(self.weights, np.dot(step_size, np.matmul(XX, self.weights))), np.dot(step_size, Xy)) # w = w − step_size*XXw + step_size*Xy
            self.weights = BatchGDLasso.prox(self, step_size, w)

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RMSprop(Regressor):

    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'step_size': 0.001, 'decay': 0.9, 'batch_size': 128,'epochs': 1000}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.random.rand(Xtrain.shape[1])#.reshape((Xtrain.shape[1], 1))
        MeanSquare = np.zeros_like(self.weights) # moving average of the squared gradient
        epsilon = 1e-8  # epsilon to be added to the denominator to avoid dividing by zero
        for i in range(self.params['epochs']):
            permutation = list(np.random.permutation(numsamples))
            shuffled_X = Xtrain[permutation, :]
            shuffled_y = ytrain[permutation]
            for j in range(0, numsamples, self.params['batch_size']):
                mini_X = shuffled_X[j:j + self.params['batch_size'], :]
                mini_y = shuffled_y[j:j + self.params['batch_size']]
                g = np.dot(mini_X.T, np.subtract(np.dot(mini_X, self.weights), mini_y)) / len(mini_y) # ∇c = x.T*(x*w − y) / n
                MeanSquare = np.add(np.dot(self.params['decay'], MeanSquare), np.dot((1-self.params['decay']), np.square(g))) # MeanSquare = decay*MeanSquare + (1-decay)* g^2
                self.weights = np.subtract(self.weights, np.divide(np.dot(self.params['step_size'], g), np.add(np.sqrt(MeanSquare), epsilon))) # w = w - ( step_size*g / (sqrt(MeanSquare) + eps) )

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest

class AMSGrad(Regressor):

    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'step_size': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'batch_size': 128,'epochs': 1000}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.random.rand(Xtrain.shape[1])#.reshape((Xtrain.shape[1], 1))
        m = np.zeros_like(self.weights)
        v = np.zeros_like(self.weights)
        v_hat = np.zeros_like(self.weights)
        epsilon = 1e-8  # epsilon to be added to the denominator to avoid dividing by zero
        for i in range(self.params['epochs']):
            permutation = list(np.random.permutation(numsamples))
            shuffled_X = Xtrain[permutation, :]
            shuffled_y = ytrain[permutation]
            for j in range(0, numsamples, self.params['batch_size']):
                mini_X = shuffled_X[j:j + self.params['batch_size'], :]
                mini_y = shuffled_y[j:j + self.params['batch_size']]
                g = np.dot(mini_X.T, np.subtract(np.dot(mini_X, self.weights), mini_y)) / len(mini_y) # ∇c = x.T*(x*w − y) / n
                m = np.add(np.dot(self.params['beta_1'], m), np.dot((1-self.params['beta_1']), g)) # m = beta_1*m + (1-beta_1)* g
                v = np.add(np.dot(self.params['beta_2'], v), np.dot((1-self.params['beta_2']), np.square(g))) # v = beta_2*v + (1-beta_2)* g^2
                v_hat = np.maximum(v_hat, v)
                self.weights = np.subtract(self.weights, np.divide(np.dot(self.params['step_size'], m), np.add(np.sqrt(v_hat), epsilon))) # w = w - ( step_size*m / (sqrt(v_hat) + epsilon)
                

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest
