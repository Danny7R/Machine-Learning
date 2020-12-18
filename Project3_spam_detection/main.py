from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# import modified_FCM as mFCM

svm_tra_acc = np.zeros((10))
svm_tst_acc = np.zeros((10))
svm_rec = np.zeros((10))
svm_prec = np.zeros((10))
svm_f1 = np.zeros((10))

tree_tra_acc = np.zeros((10))
tree_tst_acc = np.zeros((10))
tree_rec = np.zeros((10))
tree_prec = np.zeros((10))
tree_f1 = np.zeros((10))

rfor_tra_acc = np.zeros((10))
rfor_tst_acc = np.zeros((10))
rfor_rec = np.zeros((10))
rfor_prec = np.zeros((10))
rfor_f1 = np.zeros((10))

nn_tra_acc = np.zeros((10))
nn_tst_acc = np.zeros((10))
nn_rec = np.zeros((10))
nn_prec = np.zeros((10))
nn_f1 = np.zeros((10))

for i in range(10):
    df_tra = pd.read_csv(('spambase-10-fold/spambase-10-'+str(i+1)+'tra.dat'), sep=',',header=None)
    df_tst = pd.read_csv(('spambase-10-fold/spambase-10-'+str(i+1)+'tst.dat'), sep=',',header=None)
    # df_tra = df_tra.reindex(np.random.permutation(df_tra.index))
    # df_tst = df_tst.reindex(np.random.permutation(df_tst.index))
    ds_tra = df_tra.values
    ds_tst = df_tst.values
    x_train = ds_tra[:, :-1]
    y_train = ds_tra[:, -1]
    x_test = ds_tst[:, :-1]
    y_test = ds_tst[:, -1]
    print('fold '+str(i)+':', x_train.shape, x_test.shape, y_train.shape, y_test.shape, np.unique(y_train, return_counts=False), np.unique(y_test, return_counts=False))

    ## svm
    svm = SVC(kernel='rbf', gamma='auto')
    svm.fit(x_train, y_train)
    svm_tra_acc[i] = svm.score(x_train, y_train)
    svm_tst_acc[i] = svm.score(x_test, y_test)
    y_pred = svm.predict(x_test)
    svm_rec[i] = sklearn.metrics.recall_score(y_test, y_pred)
    svm_prec[i] = sklearn.metrics.precision_score(y_test, y_pred)
    svm_f1[i] = sklearn.metrics.f1_score(y_test, y_pred)

    ## tree
    tree1 = tree.DecisionTreeClassifier(max_depth=15, min_samples_leaf=5)#min_impurity_decrease=0
    tree1 = tree1.fit(x_train, y_train)
    tree_tra_acc[i] = tree1.score(x_train, y_train)
    tree_tst_acc[i] = tree1.score(x_test, y_test)
    y_pred = tree1.predict(x_test)
    tree_rec[i] = sklearn.metrics.recall_score(y_test, y_pred)
    tree_prec[i] = sklearn.metrics.precision_score(y_test, y_pred)
    tree_f1[i] = sklearn.metrics.f1_score(y_test, y_pred)

    ## Random Forest
    rforest = RandomForestClassifier(n_estimators=100)
    rforest = rforest.fit(x_train, y_train)
    rfor_tra_acc[i] = rforest.score(x_train, y_train)
    rfor_tst_acc[i] = rforest.score(x_test, y_test)
    y_pred = rforest.predict(x_test)
    rfor_rec[i] = sklearn.metrics.recall_score(y_test, y_pred)
    rfor_prec[i] = sklearn.metrics.precision_score(y_test, y_pred)
    rfor_f1[i] = sklearn.metrics.f1_score(y_test, y_pred)

    ## NN
    nn = MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=(32), alpha=0.01, max_iter=1000)
    nn = nn.fit(x_train, y_train)
    nn_tra_acc[i] = nn.score(x_train, y_train)
    nn_tst_acc[i] = nn.score(x_test, y_test)
    y_pred = nn.predict(x_test)
    nn_rec[i] = sklearn.metrics.recall_score(y_test, y_pred)
    nn_prec[i] = sklearn.metrics.precision_score(y_test, y_pred)
    nn_f1[i] = sklearn.metrics.f1_score(y_test, y_pred)

    ## fuzzy
    classes = np.unique(y_train, return_counts=False)
##    lb = sklearn.preprocessing.LabelBinarizer()
##    lb.fit(classes)
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(classes.reshape(-1, 1))
    u_init = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_train_nu = np.argmax(u_init, axis=1)
    z = np.zeros((u_init.shape[0], 2), dtype=u_init.dtype)
    u_init = np.concatenate((u_init,z), axis=1)

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(x_train.T, 4, 5, error=0.005, maxiter=1000, init=u_init.T)
    y_train_pred = np.argmax(u.T, axis=1)
    y_train_pred = y_train_pred > 1
    print('fuzzy train:', sklearn.metrics.accuracy_score(y_train_nu, y_train_pred))

    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(x_test.T, cntr, 5, error=0.005, maxiter=1000)
##    print(u.shape)
    y_test_pred = np.argmax(u.T, axis=1)
    u_test = enc.transform(y_test.reshape(-1, 1)).toarray()
    y_test_nu = np.argmax(u_test, axis=1)
    y_test_nu = y_test_nu > 1
    print('fuzzy test:', sklearn.metrics.accuracy_score(y_test_nu, y_test_pred))

print('svm train:', svm_tra_acc, 'mean:', np.mean(svm_tra_acc))
print('svm test:', svm_tst_acc, 'mean:', np.mean(svm_tst_acc))
print('recall:', svm_rec, 'mean:', np.mean(svm_rec))
print('precision:', svm_prec, 'mean:', np.mean(svm_prec))
print('f1_score:', svm_f1, 'mean:', np.mean(svm_f1))

print('###########################################')
print('tree train:', tree_tra_acc, 'mean:', np.mean(tree_tra_acc))
print('tree test:', tree_tst_acc, 'mean:', np.mean(tree_tst_acc))
print('recall:', tree_rec, 'mean:', np.mean(tree_rec))
print('precision:', tree_prec, 'mean:', np.mean(tree_prec))
print('f1_score:', tree_f1, 'mean:', np.mean(tree_f1))

print('###########################################')
print('rforest train:', rfor_tra_acc, 'mean:', np.mean(rfor_tra_acc))
print('rforest test:', rfor_tst_acc, 'mean:', np.mean(rfor_tst_acc))
print('recall:', rfor_rec, 'mean:', np.mean(rfor_rec))
print('precision:', rfor_prec, 'mean:', np.mean(rfor_prec))
print('f1_score:', rfor_f1, 'mean:', np.mean(rfor_f1))

print('###########################################')
print('nn train:', nn_tra_acc, 'mean:', np.mean(nn_tra_acc))
print('nn test:', nn_tst_acc, 'mean:', np.mean(nn_tst_acc))
print('recall:', nn_rec, 'mean:', np.mean(nn_rec))
print('precision:', nn_prec, 'mean:', np.mean(nn_prec))
print('f1_score:', nn_f1, 'mean:', np.mean(nn_f1))

import scipy
print(scipy.stats.ttest_ind(nn_tst_acc, rfor_tst_acc))
