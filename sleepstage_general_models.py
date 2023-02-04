# saving firstmodels

import numpy as np
from sklearn import  preprocessing
import models
scaler = preprocessing.MinMaxScaler()
import pickle
import random
from scipy.io import loadmat

all_num = np.array([35, 45, 55, 65, 83, 104, 106, 108, 110, 112, 126, 137, 151, 154, 163, 26, 53, 105,
                    1, 12, 14, 20, 41, 62, 77, 133, 153, 15, 42, 44, 64, 89, 160, 161,
                    25, 47, 50, 56, 59, 63, 92, 111, 51, 58, 94, 102, 114, 144, 122, 38, 68, 91, 95, 98, 138,
                    2, 4, 96, 118, 125, 142, 21, 67, 101, 115, 127, 136, 99, 116, 130, 11, 49, 82,
                    24, 28, 31, 34, 36, 40, 60, 76, 128, 135, 57, 66, 71, 79, 109, 139, 152,
                    30, 43, 46, 69, 70, 73, 75, 86, 97, 117, 129, 131, 134, 156, 162, 19, 23, 74, 78, 84, 85])
control_num = all_num
len(control_num)

num = 0
train_set = np.empty((48, 0))
train_set_label = np.empty((1, 0))

for m in control_num:
    train_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject{}_PSD_TOTAL.mat".format(m))
    train_data = train_data['TOTAL']
    train_set = np.append(train_set, train_data, 1)

    train_label = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject{}_PSD_TOTAL_label.mat".format(m))
    train_label = train_label['label_TOTAL']
    train_set_label = np.append(train_set_label, train_label, 1)

ind1 = np.where(train_set_label == 1)
ind2 = np.where(train_set_label == 2)
ind3 = np.where(train_set_label == 3)
ind4 = np.where(train_set_label == 4)
ind5 = np.where(train_set_label == 5)
print([np.size(ind1[1]), np.size(ind2[1]), np.size(ind3[1]), np.size(ind4[1]), np.size(ind5[1])])
st_min = np.min([np.size(ind1[1]), np.size(ind2[1]), np.size(ind3[1]), np.size(ind5[1])])
st1 = list(range(np.size(ind1[1])));
st1_pos = random.sample(st1, st_min);
st1_p = ind1[1][st1_pos]
st2 = list(range(np.size(ind2[1])));
st2_pos = random.sample(st2, st_min);
st2_p = ind2[1][st2_pos]
st3 = list(range(np.size(ind3[1])));
st3_pos = random.sample(st3, st_min);
st3_p = ind3[1][st3_pos]
st4_p = ind4[1]
st5 = list(range(np.size(ind5[1])));
st5_pos = random.sample(st5, st_min);
st5_p = ind5[1][st5_pos]

train_set2 = np.hstack((train_set[:, st1_p], train_set[:, st2_p], train_set[:, st3_p], train_set[:, st4_p], train_set[:, st5_p]))
train_set_label2 = np.hstack((train_set_label[:, st1_p], train_set_label[:, st2_p], train_set_label[:, st3_p], train_set_label[:, st4_p], train_set_label[:, st5_p]))

train_set2 = np.transpose(train_set2)
train_set_label2 = np.transpose(train_set_label2).ravel()

# svm
general_model_svm = models.SVM_train(train_set2, train_set_label2)
filename = "/user/chae2089/CWpaper/FN_REVISION/GENERAL_MODEL_num_stage_fix/SVM/GENERAL_MODEL_SVM2.sav"
pickle.dump(general_model_svm, open(filename, 'wb'))

# knn
general_model_knn = models.kNN_train(train_set2, train_set_label2)
filename = "/user/chae2089/CWpaper/FN_REVISION/GENERAL_MODEL_num_stage_fix/kNN/GENERAL_MODEL_kNN.sav"
pickle.dump(general_model_knn, open(filename, 'wb'))

# mlp
general_model_mlp = models.MLP_train(train_set, train_set_label2)
filename = "/user/chae2089/CWpaper/FN_REVISION/GENERAL_MODEL/MLP/GENERAL_MODEL_MLP.sav"
pickle.dump(general_model_mlp, open(filename, 'wb'))

