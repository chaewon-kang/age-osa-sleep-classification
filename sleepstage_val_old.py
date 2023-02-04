# Sleep stage classification old model - SVM, kNN, MLP

import numpy as np
import random
import models
from scipy.io import loadmat

# create 5-fold cross validation x 10 sets indices
p_num=55
a = []
for i in range(10):
    a.append(np.ceil((np.random.permutation(p_num)+1)/(p_num/5.0)).reshape(1,p_num))
a = np.array(a)

# divide groups to young and old
normal_Y=np.array([35,45,55,65,83,104,106,108,110,112,126,137,151,154,163,26,53,105])
normal_O=np.array([1,12,14,20,41,62,77,133,153,15,42,44,64,89,160,161])
mtom_Y=np.array([25,47,50,56,59,63,92,111,51,58,94,102,114,144,122,38,68,91,95,98,138])
mtom_O=np.array([2,4,96,118,125,142,21,67,101,115,127,136,99,116,130,11,49,82])
severe_Y=np.array([24,28,31,34,36,40,60,76,128,135,57,66,71,79,109,139,152])
severe_O=np.array([30,43,46,69,70,73,75,86,97,117,129,131,134,156,162,19,23,74,78,84,85])

young = np.concatenate((normal_Y, mtom_Y, severe_Y), axis=None)
old = np.concatenate((normal_O, mtom_O, severe_O), axis=None)

# 5-cross validation
num = 0
for k in range(len(a)):
    b = a[k]

    for m in range(1, 6):
        index_test = np.where(b == m)
        index_train = np.where(b != m)

        test_set = np.empty((48, 0))
        train_set = np.empty((48, 0))
        test_set_label = np.empty((1, 0))
        train_set_label = np.empty((1, 0))

        test_set_Y = np.empty((48, 0))
        test_set_label_Y = np.empty((1, 0))

        train_set2 = np.empty((48, 0))
        test_set_label2 = np.empty((1, 0))

        test = index_test[1]
        train = index_train[1]

        print("{}th outer loop's {}th test index".format(k, m))
        print(test)
        print("{}th outer loop's {}th train index".format(k, m))
        print(train)
        print("\n")

        # O-O: older group validation set
        for n in test:
            test_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject{}_PSD_TOTAL.mat".format(old[n]))
            test_data = test_data['TOTAL']
            test_set = np.append(test_set, test_data, 1)

            test_label = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject{}_PSD_TOTAL_label.mat".format(old[n]))
            test_label = test_label['label_TOTAL']
            test_set_label = np.append(test_set_label, test_label, 1)
        test_set = np.transpose(test_set)
        test_set_label = np.transpose(test_set_label).ravel()

        # O-Y: younger group validation set
        c_1 = random.sample(range(len(young)), len(test))
        for n in young[c_1]:
            test_data_Y = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject{}_PSD_TOTAL.mat".format(n))
            test_data_Y = test_data_Y['TOTAL']
            test_set_Y = np.append(test_set_Y, test_data_Y, 1)

            test_label_Y = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject{}_PSD_TOTAL_label.mat".format(n))
            test_label_Y = test_label_Y['label_TOTAL']
            test_set_label_Y = np.append(test_set_label_Y, test_label_Y, 1)
        test_set_Y = np.transpose(test_set_Y)
        test_set_label_Y = np.transpose(test_set_label_Y).ravel()

        # read train dataset
        for m in train:
            train_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject{}_PSD_TOTAL.mat".format(old[m]))
            train_data = train_data['TOTAL']
            train_set = np.append(train_set, train_data, 1)

            train_label = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject{}_PSD_TOTAL_label.mat".format(old[m]))
            train_label = train_label['label_TOTAL']
            train_set_label = np.append(train_set_label, train_label, 1)

        ind1 = np.where(train_set_label == 1)
        ind2 = np.where(train_set_label == 2)
        ind3 = np.where(train_set_label == 3)
        ind4 = np.where(train_set_label == 4)
        ind5 = np.where(train_set_label == 5)
        print([np.size(ind1[1]), np.size(ind2[1]), np.size(ind3[1]), np.size(ind4[1]), np.size(ind5[1])])
        st_min = np.min([np.size(ind1[1]), np.size(ind2[1]), np.size(ind3[1]), np.size(ind5[1])])
        # st_num=np.where([np.size(ind1[1]),np.size(ind2[1]),np.size(ind3[1]),np.size(ind5[1])]==st_min)[0]
        # print(st_num+1)
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

        # redefined train set
        train_set2 = np.hstack((train_set[:, st1_p], train_set[:, st2_p], train_set[:, st3_p], train_set[:, st4_p], train_set[:, st5_p]))
        train_set_label2 = np.hstack((train_set_label[:, st1_p], train_set_label[:, st2_p], train_set_label[:, st3_p], train_set_label[:, st4_p], train_set_label[:, st5_p]))

        train_set2 = np.transpose(train_set2)
        train_set_label2 = np.transpose(train_set_label2).ravel()

        # SVM
        model_s = models.SVM_train(train_set2, train_set_label2)
        # model prediction: O-O
        fname_report = ("/user/chae2089/CWpaper/FN_REVISION/KFOLD_num_stage_fix/SVM/old/old_test/result_{}".format(num))
        fname_result = ("/user/chae2089/CWpaper/FN_REVISION/KFOLD_num_stage_fix/SVM/old/old_test/report_{}.csv".format(num))
        models.val_test(model_s, test_set, test_set_label, fname_result, fname_report)
        # model prediction: O-Y
        fname_report = ("/user/chae2089/CWpaper/FN_REVISION/KFOLD_num_stage_fix/SVM/old/young_test/result_{}".format(num))
        fname_result = ("/user/chae2089/CWpaper/FN_REVISION/KFOLD_num_stage_fix/SVM/old/young_test/report_{}.csv".format(num))
        models.val_test(model_s, test_set_Y, test_set_label_Y, fname_result, fname_report)

        # kNN
        model_k = models.kNN_train(train_set2, train_set_label2)
        # model prediction: O-O
        fname_report = ("/user/chae2089/CWpaper/FN_REVISION/KFOLD_num_stage_fix/kNN/old/old_test/result_{}".format(num))
        fname_result = ("/user/chae2089/CWpaper/FN_REVISION/KFOLD_num_stage_fix/kNN/old/old_test/report_{}.csv".format(num))
        models.val_test(model_k, test_set, test_set_label, fname_result, fname_report)
        # model prediction: O-Y
        fname_report = ("/user/chae2089/CWpaper/FN_REVISION/KFOLD_num_stage_fix/kNN/old/young_test/result_{}".format(num))
        fname_result = ("/user/chae2089/CWpaper/FN_REVISION/KFOLD_num_stage_fix/kNN/old/young_test/report_{}.csv".format(num))
        models.val_test(model_k, test_set_Y, test_set_label_Y, fname_result, fname_report)

        # MLP
        model_m = models.SVM_train(train_set2, train_set_label2)
        # model prediction: O-O
        fname_report = ("/user/chae2089/CWpaper/FN_REVISION/KFOLD_num_stage_fix/MLP/old/old_test/result_{}".format(num))
        fname_result = ("/user/chae2089/CWpaper/FN_REVISION/KFOLD_num_stage_fix/MLP/old/old_test/report_{}.csv".format(num))
        models.val_test(model_m, test_set, test_set_label, fname_result, fname_report)
        # model prediction: O-Y
        fname_report = ("/user/chae2089/CWpaper/FN_REVISION/KFOLD_num_stage_fix/kNN/old/young_test/result_{}".format(num))
        fname_result = ("/user/chae2089/CWpaper/FN_REVISION/KFOLD_num_stage_fix/kNN/old/young_test/report_{}.csv".format(num))
        models.val_test(model_m, test_set_Y, test_set_label_Y, fname_result, fname_report)