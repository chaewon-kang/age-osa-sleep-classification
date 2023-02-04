import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

import models

scaler = preprocessing.MinMaxScaler()
import pickle

normal_Y_test = np.array([29, 90, 119, 140])
normal_O_test = np.array([164,165,167,168])
mtom_Y_test = np.array([54, 80,88,107])
mtom_O_test= np.array([16,72,93,103,121,145])
severe_Y_test = np.array([132,175,176,177,179])
severe_O_test = np.array([52,169,170,171,174])
test_num = np.array([normal_O_test, normal_Y_test, mtom_O_test, mtom_Y_test, severe_O_test, severe_Y_test])


# svm
osa_svm = pickle.load(open("/user/chae2089/CWpaper/FN_REVISION/GENERAL_2nd_MODEL/SVM/GENERAL_2nd_MODEL_SVM_only_H_S.sav", 'rb'))
normal_y = models.osa_test(osa_svm, "SVM", normal_Y_test)
normal_o = models.osa_test(osa_svm, "SVM", normal_O_test)
mtom_y = models.osa_test(osa_svm, "SVM", mtom_Y_test)
mtom_o = models.osa_test(osa_svm, "SVM", mtom_O_test)
severe_y = models.osa_test(osa_svm, "SVM", severe_Y_test)
severe_o = models.osa_test(osa_svm, "SVM", severe_O_test)
models.osa_test_score("svm", normal_y, normal_o, mtom_y, mtom_o, severe_y, severe_o)

# kNN
osa_knn = pickle.load(open("/user/chae2089/CWpaper/FN_REVISION/GENERAL_2nd_MODEL/kNN/GENERAL_2nd_MODEL_kNN_only_H_S.sav", 'rb'))
normal_y = models.osa_test(osa_knn, "kNN", normal_Y_test)
normal_o = models.osa_test(osa_knn, "kNN", normal_O_test)
mtom_y = models.osa_test(osa_knn, "kNN", mtom_Y_test)
mtom_o = models.osa_test(osa_knn, "kNN", mtom_O_test)
severe_y = models.osa_test(osa_knn, "kNN", severe_Y_test)
severe_o = models.osa_test(osa_knn, "kNN", severe_O_test)
models.osa_test_score("kNN", normal_y, normal_o, mtom_y, mtom_o, severe_y, severe_o)

# MLP
osa_mlp = pickle.load(open("/user/chae2089/CWpaper/FN_REVISION/GENERAL_2nd_MODEL/MLP/GENERAL_2nd_MODEL_MLP_only_H_S.sav", 'rb'))
normal_y = models.osa_test(osa_mlp, "MLP", normal_Y_test)
normal_o = models.osa_test(osa_mlp, "MLP", normal_O_test)
mtom_y = models.osa_test(osa_mlp, "MLP", mtom_Y_test)
mtom_o = models.osa_test(osa_mlp, "MLP", mtom_O_test)
severe_y = models.osa_test(osa_mlp, "MLP", severe_Y_test)
severe_o = models.osa_test(osa_mlp, "MLP", severe_O_test)
models.osa_test_score("MLP", normal_y, normal_o, mtom_y, mtom_o, severe_y, severe_o)







