# OSA classification general model - SVM, kNN, MLP

import numpy as np
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
from scipy.io import loadmat
import models

p_num=111
a = []
for i in range(10):
    a.append(np.ceil((np.random.permutation(p_num)+1)/(p_num/5.0)).reshape(1,p_num))
a = np.array(a)
len(a)

normal=np.array([35,45,55,65,83,104,106,108,110,112,126,137,151,154,163,26,53,105,
                 1,12,14,20,41,62,77,133,153,15,42,44,64,89,160,161])
mtom=np.array([25,47,50,56,59,63,92,111,51,58,94,102,114,144,122,38,68,91,95,98,138,
               2,4,96,118,125,142,21,67,101,115,127,136,99,116,130,11,49,82])
severe=np.array([24,28,31,34,36,40,60,76,128,135,57,66,71,79,109,139,152,
                 30,43,46,69,70,73,75,86,97,117,129,131,134,156,162,19,23,74,78,84,85])
all_num = np.concatenate((normal, mtom, severe), axis=None)

control_num = all_num
len(control_num)

num=0
feature_num=97
for k in range(len(a)):
    b = a[k]

    for m in range(1, 6):
        index_test = np.where(b == m)
        index_train = np.where(b != m)

        test_set = np.empty((0, feature_num))
        train_set = np.empty((0, feature_num))
        test_set_label = np.empty((0, 1))
        train_set_label = np.empty((0, 1))
        train_set2 = np.empty((0, feature_num))
        test_set_label2 = np.empty((0, 1))

        test = index_test[1]
        train = index_train[1]

        print("{}th outer loop's {}th test index".format(k, m))
        print(test)
        print("{}th outer loop's {}th train index".format(k, m))
        print(train)
        print("\n")

        # Toggle 1)groups 2)SpO2 component
        for n in test:
            test_label = np.empty((0, 1))
            if np.isin(normal, control_num[n]).sum() == 1:
                test_label = np.array([[1]])
                test_data = loadmat("/user/chae2089/CWpaper/FINAL/2nd_model/feature_nrem_rem_str/feature/subject{}.mat".format(control_num[n]))
                # Testing with SpO2
                # test_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_2nd_model/eeg_combined_spo2/subject{}.mat".format(control_num[n]))
                test_data = np.transpose(test_data['nrem_rem_comb_spo2'])
                test_set = np.append(test_set, test_data, 0)
                test_set_label = np.append(test_set_label, test_label, 0)

            # Including mtom class
            # elif np.isin(mtom,control_num[n]).sum() == 1:
            #      test_label = np.array([[2]])
            #      test_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_2nd_model/eeg_combined_spo2/subject{}.mat".format(control_num[n]))
            #      #test_data = loadmat("/user/chae2089/CWpaper/FINAL/2nd_model/feature_nrem_rem_str/feature/subject{}.mat".format(control_num[n]))
            #      test_data = np.transpose(test_data['nrem_rem_comb_spo2'])
            #      test_set = np.append(test_set, test_data, 0)
            #      test_set_label = np.append(test_set_label, test_label, 0)

            elif np.isin(severe, control_num[n]).sum() == 1:
                test_label = np.array([[2]])
                test_data = loadmat("/user/chae2089/CWpaper/FINAL/2nd_model/feature_nrem_rem_str/feature/subject{}.mat".format(control_num[n]))
                # Testing with SpO2
                #test_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_2nd_model/eeg_combined_spo2/subject{}.mat".format(control_num[n]))
                test_data = np.transpose(test_data['nrem_rem_comb_spo2'])
                test_set = np.append(test_set, test_data, 0)
                test_set_label = np.append(test_set_label, test_label, 0)

        # Toggle 1)groups 2)SpO2 component
        for m in train:
            train_label = np.empty((0, 1))
            if np.isin(normal, control_num[m]).sum() == 1:
                train_label = np.array([[1]])
                train_data = loadmat("/user/chae2089/CWpaper/FINAL/2nd_model/feature_nrem_rem_str/feature/subject{}".format(control_num[m]))
                # Train with SpO2
                #train_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_2nd_model/eeg_combined_spo2/subject{}".format(control_num[m]))
                train_data = np.transpose(train_data['nrem_rem_comb_spo2'])
                train_set = np.append(train_set, train_data, 0)
                train_set_label = np.append(train_set_label, train_label, 0)

            # Including mtom class
            # elif np.isin(mtom,control_num[m]).sum() == 1:
            #     train_label = np.array([[2]])
            #     train_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_2nd_model/eeg_combined_spo2/subject{}".format(control_num[m]))
            #     train_data = loadmat("/user/chae2089/CWpaper/FINAL/2nd_model/feature_nrem_rem_str/feature/subject{}".format(control_num[m]))
            #     train_data = np.transpose(train_data['nrem_rem_comb_spo2'])
            #     train_set = np.append(train_set, train_data, 0)
            #     train_set_label = np.append(train_set_label, train_label, 0)

            elif np.isin(severe, control_num[m]).sum() == 1:
                train_label = np.array([[2]])
                train_data = loadmat("/user/chae2089/CWpaper/FINAL/2nd_model/feature_nrem_rem_str/feature/subject{}".format(control_num[m]))
                # Train with SpO2
                #train_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_2nd_model/eeg_combined_spo2/subject{}".format(control_num[m]))
                train_data = np.transpose(train_data['nrem_rem_comb_spo2'])
                train_set = np.append(train_set, train_data, 0)
                train_set_label = np.append(train_set_label, train_label, 0)

        train_set2 = train_set
        train_set_label2 = train_set_label.ravel()

        test_set = np.transpose(test_set)
        test_set_label = np.transpose(test_set_label)

        # SVM
        model_s = models.SVM_train(train_set2, train_set_label2)
        score = models.val_test_osa(model_s, test_set, test_set_label)
        print("osa_general_svm result = {}".format(score[0]))
        print("osa_general_svm accuracy = {}".format(score[1]))

        # kNN
        model_k = models.kNN_train(train_set2, train_set_label2)
        score = models.val_test_osa(model_k, test_set, test_set_label)
        print("osa_general_kNN result = {}".format(score[0]))
        print("osa_general_kNN accuracy = {}".format(score[1]))

        # MLP
        model_m = models.MLP_train(train_set2, train_set_label2)
        score = models.val_test_osa(model_m, test_set, test_set_label)
        print("osa_general_MLP result = {}".format(score[0]))
        print("osa_general_MLP accuracy = {}".format(score[1]))

        num = num + 1






