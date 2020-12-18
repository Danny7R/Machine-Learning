from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import time
import dataloader as dtl
import regressionalgorithms as algs

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]


if __name__ == '__main__':
    trainsize = 1000
    testsize = 5000
    numruns = 1

    regressionalgs = {'Random': algs.Regressor(),
##                'Mean': algs.MeanPredictor(),
##                'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
##                'FSLinearRegression50': algs.FSLinearRegression({'features': range(50)}),
##                'RidgeLinearRegression': algs.RidgeLinearRegression(),
                'SGD': algs.SGD({'step_size': 0.01, 'epochs': 1000, 'Report_error_graph': True}),
##                'BatchGD': algs.BatchGD({'step_size_max': 1.0, 'max_iterations': 1000}),
##                'BatchGDLasso': algs.BatchGDLasso({'reg': 0.01, 'max_iterations': 1000}),
##                'RMSprop': algs.RMSprop({'step_size': 0.001, 'decay': 0.9, 'batch_size': 128,'epochs': 1000}),
##                'AMSGrad': algs.AMSGrad({'step_size': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'batch_size': 128,'epochs': 1000}),
             }
    numalgs = len(regressionalgs)

    # Enable the best parameter to be selected, to enable comparison
    # between algorithms with their best parameter settings
    parameters = (
##        {'regwgt': 0.0},
        {'regwgt': 0.01},
##        {'regwgt': 1.0},
                      )
    numparams = len(parameters)
    
    errors = {}
    times = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))
        times[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_ctscan(trainsize,testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in regressionalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                t0 = 1000*time.time()
                learner.learn(trainset[0], trainset[1])
                t1 = 1000*time.time()
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                print ('Training time for ', learnername, ': ', t1-t0, '(ms)')
                errors[learnername][p,r] = error
                times[learnername][p,r] = t1-t0


    for learnername in regressionalgs:
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p
        avetime = np.mean(times[learnername][bestparams,:])

        # Extract best parameters
        learner.reset(parameters[bestparams])
        #print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror))
        print ('Standard error for ' + learnername + ': ' + str(np.std(errors[learnername][bestparams,:])/np.sqrt(numruns)))
        print ('Average training time for ' + learnername + ': ', avetime, '(ms)')
