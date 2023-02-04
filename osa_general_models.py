# saving osa classification models

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

import models

scaler = preprocessing.MinMaxScaler()
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.io import loadmat, savemat
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

normal=np.array([35,45,55,65,83,104,106,108,110,112,126,137,151,154,163,26,53,105,
                 1,12,14,20,41,62,77,133,153,15,42,44,64,89,160,161])
mtom=np.array([25,47,50,56,59,63,92,111,51,58,94,102,114,144,122,38,68,91,95,98,138,
               2,4,96,118,125,142,21,67,101,115,127,136,99,116,130,11,49,82])
severe=np.array([24,28,31,34,36,40,60,76,128,135,57,66,71,79,109,139,152,
                 30,43,46,69,70,73,75,86,97,117,129,131,134,156,162,19,23,74,78,84,85])
all_num = np.concatenate((normal, mtom, severe), axis=None)

control_num = all_num
len(control_num)

for m in control_num:

    train_label = np.empty((0, 1))
    if np.isin(normal, m).sum() == 1:
        train_label = np.array([[1]])
        train_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_2nd_model/eeg_alone/subject{}".format(m))
        train_data = np.transpose(train_data['nrem_rem'])
        train_set = np.append(train_set, train_data, 0)
        train_set_label = np.append(train_set_label, train_label, 0)

    #     elif np.isin(mtom,m).sum() == 1:
    #         train_label = np.array([[2]])
    #         train_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_2nd_model/eeg_alone/subject{}".format(m))
    #         train_data = np.transpose(train_data['nrem_rem'])
    #         train_set = np.append(train_set, train_data, 0)
    #         train_set_label = np.append(train_set_label, train_label, 0)

    elif np.isin(severe, m).sum() == 1:
        train_label = np.array([[2]])
        train_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_2nd_model/eeg_alone/subject{}".format(m))
        train_data = np.transpose(train_data['nrem_rem'])
        train_set = np.append(train_set, train_data, 0)
        train_set_label = np.append(train_set_label, train_label, 0)

train_set2 = train_set
train_set_label2 = train_set_label.ravel()

# svm
osa_model_svm = models.SVM_train(train_set2, train_set_label2)
filename = "/user/chae2089/CWpaper/FN_REVISION/GENERAL_2nd_MODEL/SVM/GENERAL_2nd_MODEL_SVM_only_H_S.sav"
pickle.dump(osa_model_svm, open(filename, 'wb'))
#filename = "/user/chae2089/CWpaper/FN_REVISION/GENERAL_2nd_MODEL/SVM/GENERAL_2nd_MODEL_SVM.sav"
#pickle.dump(osa_model_svm, open(filename, 'wb'))

# knn
osa_model_knn = models.kNN_train(train_set2, train_set_label2)
filename = "/user/chae2089/CWpaper/FN_REVISION/GENERAL_2nd_MODEL/kNN/GENERAL_2nd_MODEL_kNN_only_H_S.sav"
pickle.dump(osa_model_knn, open(filename, 'wb'))
#filename = "/user/chae2089/CWpaper/FN_REVISION/GENERAL_2nd_MODEL/kNN/GENERAL_2nd_MODEL_kNN.sav"
#pickle.dump(osa_model_knn, open(filename, 'wb'))

# mlp
osa_model_mlp = models.mlp_train(train_set2, train_set_label2)
filename = "/user/chae2089/CWpaper/FN_REVISION/GENERAL_2nd_MODEL/MLP/GENERAL_2nd_MODEL_MLP_only_H_S.sav"
pickle.dump(osa_model_mlp, open(filename, 'wb'))
#filename = "/user/chae2089/CWpaper/FN_REVISION/GENERAL_2nd_MODEL/MLP/GENERAL_2nd_MODEL_MLP.sav"
#pickle.dump(osa_model_mlp, open(filename, 'wb'))

