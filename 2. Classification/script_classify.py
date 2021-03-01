from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import dataloader as dtl
import classalgorithms as algs


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))

## k-fold cross-validation
# K - number of folds
# X - data to partition
# Y - targets to partition
# classalgs - a dictionary mapping algorithm names to algorithm instances
#
# example:
# classalgs = {
#   'nn_0.01': algs.NeuralNet({ 'regwgt': 0.01 }),
#   'nn_0.1':  algs.NeuralNet({ 'regwgt': 0.1  }),
# }

def cross_validate(K, X, Y, classalgs, stratify=False):
    print(classalgs.keys())
    errs = {}
    for learnername, learner in classalgs.items():
        errs[learnername] = np.zeros((K))
        for k in range(K):
            if (stratify == True):
                x0 = X[Y == 0]
                x1 = X[Y == 1]
                y0 = Y[Y == 0]
                y1 = Y[Y == 1]
                fold_len0 = int(x0.shape[0] / K)
                fold_len1 = int(x1.shape[0] / K)
                test_inx0 = list(range(k*fold_len0, (k+1)*fold_len0))
                test_inx1 = list(range(k*fold_len1, (k+1)*fold_len1))
                xtest = np.concatenate((x0[test_inx0], x1[test_inx1]))
                ytest = np.concatenate((y0[test_inx0], y1[test_inx1]))
                train_inx0 = list(set(list(range(x0.shape[0]))) - set(test_inx0))
                train_inx1 = list(set(list(range(x1.shape[0]))) - set(test_inx1))
                xtrain = np.concatenate((x0[train_inx0], x1[train_inx1]))
                ytrain = np.concatenate((y0[train_inx0], y1[train_inx1]))
            else:
                fold_len = int(X.shape[0] / K)
                test_inx = list(range(k*fold_len, (k+1)*fold_len))
                xtest = X[test_inx]
                ytest = Y[test_inx]
                train_inx = list(set(list(range(X.shape[0]))) - set(test_inx))
                xtrain = X[train_inx]
                ytrain = Y[train_inx]

            print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
            # Train model
            learner.learn(xtrain, ytrain)
            # Test model
            predictions = learner.predict(xtest)
            err = geterror(ytest, predictions)
            print('Error for ' + learnername + ': ' + str(err))
            errs[learnername][k] = err
    
    besterr = 120
    
    for learnername, learner in classalgs.items():
        print('std error for ' + learnername + ': ' + str(np.std(errs[learnername])/np.sqrt(K)))
        aveerr = np.mean(errs[learnername])
        print('Average error for ' + learnername + ': ' + str(aveerr))
        if aveerr < besterr:
            besterr = aveerr
            best_algorithm = learnername
    print('best algorithm is: ', best_algorithm, 'with error: ', besterr)
    return best_algorithm


if __name__ == '__main__':
    trainsize = 5000
    testsize = 5000
    numruns = 1

    classalgs = {'Random': algs.Classifier(),
                 'Linear Regression': algs.LinearRegressionClass(),
                 'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 'Naive Bayes Ones': algs.NaiveBayes({'usecolumnones': True}),
                 'Linear Regression': algs.LinearRegressionClass(),
                 'Logistic Regression': algs.LogitReg(),
                 'Neural Network': algs.NeuralNet({'epochs': 100}),
                 'Neural Network 2': algs.NeuralNet2({'epochs': 100}),
                 'KernelLogitReg': algs.KernelLogitReg(),
                #  'KernelLogitReg': algs.KernelLogitReg({'kernel': 'hamming'})

                ##################### for calling cross validation use these instead ####################
                
                #  'Neural Network 4 32': algs.NeuralNet({'nh': 4, 'batch_size': 32}),
                #  'Neural Network 16 32': algs.NeuralNet({'nh': 16, 'batch_size': 32}),
                #  'Neural Network 4 128': algs.NeuralNet({'nh': 4, 'batch_size': 128}),
                #  'Neural Network 16 128': algs.NeuralNet({'nh': 16, 'batch_size': 128}),
                #  'Logistic Regression 100 32': algs.LogitReg({'epochs': 100, 'batch_size': 32}),
                #  'Logistic Regression 1000 32': algs.LogitReg({'epochs': 1000, 'batch_size': 32}),
                #  'Logistic Regression 100 128': algs.LogitReg({'epochs': 100, 'batch_size': 128}),
                #  'Logistic Regression 1000 128': algs.LogitReg({'epochs': 1000, 'batch_size': 128})
                }
    numalgs = len(classalgs)

    parameters = (
        {'regwgt': 0.0, 'nh': 4},
        # {'regwgt': 0.01, 'nh': 8},
        # {'regwgt': 0.05, 'nh': 16},
        # {'regwgt': 0.1, 'nh': 32},
                      )
    numparams = len(parameters)

    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_susy(trainsize,testsize)
        # trainset, testset = dtl.load_susy_complete(trainsize,testsize)
        # trainset, testset = dtl.load_census(trainsize,testsize)

    # cross_validate(5, trainset[0], trainset[1], classalgs)
    # cross_validate(5, trainset[0], trainset[1], classalgs, stratify=True)

        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error


    for learnername, learner in classalgs.items():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(np.std(errors[learnername][bestparams,:])/math.sqrt(numruns)))
