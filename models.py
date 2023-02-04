import numpy as np
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
from scipy.io import loadmat
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def SVM_train(train_set2, train_set_label2):
    # gridsearch for the best parameters in SVM
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 10]}
    clf_grid = GridSearchCV(SVC(kernel='rbf'), param_grid)
    clf_grid.fit(train_set2, train_set_label2)
    C_best = clf_grid.best_params_["C"]
    gamma_best = clf_grid.best_params_["gamma"]
    # print(C_best, gamma_best)

    # train model
    model = SVC(kernel='rbf', C=C_best, gamma=gamma_best)
    model.fit(train_set2, train_set_label2)

    return model

def kNN_train(train_set2, train_set_label2):
    # gridsearch for the best parameters in kNN
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
    clf_grid = GridSearchCV(KNeighborsClassifier(), param_grid)
    clf_grid.fit(train_set2, train_set_label2)
    n_neighbors_best = clf_grid.best_params_["n_neighbors"]
    weights_best = clf_grid.best_params_["weights"]
    metric_best = clf_grid.best_params_["metric"]
    # print(n_neighbors_best, weights_best, metric_best)

    # train model
    model = KNeighborsClassifier(n_neighbors=n_neighbors_best, weights=weights_best, metric=metric_best)
    model.fit(train_set2, train_set_label2)

    return model

def MLP_train(train_set2, train_set_label2):
    # gridsearch for the best parameters in MLP
    param_grid = {'hidden_layer_sizes': [(15,), (25,), (35,), (45,), (55,),
                                         (15, 15,), (25, 15,), (35, 15,), (45, 15,), (55, 15,),
                                         (15, 25,), (25, 25,), (35, 25,), (45, 25,), (55, 25,),
                                         (15, 35,), (25, 35,), (35, 35,), (45, 35,), (55, 35,),
                                         (15, 45,), (25, 45,), (35, 45,), (45, 45,), (55, 45,),
                                         (15, 55,), (25, 55,), (35, 55,), (45, 55,), (55, 55,), ]}
    clf_grid = GridSearchCV(MLPClassifier(max_iter=200, activation='relu', solver='adam', random_state=1), param_grid)
    clf_grid.fit(train_set2, train_set_label2)
    hidden_layer_sizes_best = clf_grid.best_params_["hidden_layer_sizes"]
    #print(hidden_layer_sizes_best)

    # train model
    model = MLPClassifier(max_iter=200, activation='relu', solver='adam', random_state=1,
                          hidden_layer_sizes=hidden_layer_sizes_best)
    model.fit(train_set2, train_set_label2)

    return model

def val_test(model, test_set, test_set_label, fname_result, fname_report):
    # prediction of the validation set
    pred = model.predict(test_set)
    result = confusion_matrix(test_set_label, pred)
    np.savetxt(fname_result, result)        # save results

    # classification report
    report = classification_report(test_set_label, pred)
    report2 = report.split()
    with open(fname_report, 'w') as fp:
        for item in report2:
            fp.write("%s\n" % item)

def val_test_osa(model, test_set, test_set_label):
    result2 = np.zeros((2, 2))
    accuracy2 = []

    # prediction result
    pred = model.predict(test_set)
    result = confusion_matrix(test_set_label, pred)
    result2 = result2 + result

    # accuracy score
    accuracy = np.round(accuracy_score(test_set_label, pred), 4)
    accuracy2.append(accuracy)

    return result2, accuracy2

def general_test(model, test_data):
    for k in test_data:
        test_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/results_from_1st_model/subject{}_PSD_TOTAL.mat".format(k))
        test_data = test_data['TOTAL']
        test_data = np.transpose(test_data)

        test_label = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/results_from_1st_model/subject{}_PSD_TOTAL_label.mat".format(k))
        test_label = test_label['label_TOTAL']
        test_label = np.transpose(test_label).ravel()

        pred = model.predict(test_data)

        np.savetxt('/user/chae2089/CWpaper/FINAL_GENERAL_MODEL_stage_num_fix/pred_stage/pred_{}.csv'.format(k), pred, delimiter=";")

        result = confusion_matrix(test_label, pred)
        filename = ("/user/chae2089/CWpaper/FINAL_GENERAL_MODEL_stage_num_fix/pred_stage/result_{}".format(k))
        np.savetxt(filename, result)

        report = classification_report(test_label, pred)
        report2 = report.split()
        with open("/user/chae2089/CWpaper/FINAL_GENERAL_MODEL_stage_num_fix/pred_stage/report_{}".format(k), 'w') as fp:
            for item in report2:
                fp.write("%s\n" % item)


def osa_test(model, model_type, test_data):
    result = np.empty((0,1))
    for k in test_data:
        test_data = loadmat("/user/chae2089/CWpaper/FN_REVISION/feature_for_2nd_model/results_from_1st_model/{}/subject{}.mat".format(model_type, k))
        test_data = test_data['nrem_rem']
        test_data = np.transpose(test_data)

        pred = model.predict(test_data)

        result = np.append(result, pred.reshape(1,1), 0)

        return result


def osa_test_score(model_type, normal_y, normal_o, mtom_y, mtom_o, severe_y, severe_o):
    #clin_label = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])
    #model_result = np.hstack((normal_y.transpose(), normal_y.transpose(), mtom_y.transpose(), mtom_o.transpose(), severe_y.transpose(), severe_o.transpose()))
    # print(np.vstack((clin_label,model_result)))

    TP = (np.where(mtom_y == 2)[0]).shape[0] + (np.where(mtom_o == 2)[0]).shape[0] + (np.where(severe_y == 2)[0]).shape[0] + (np.where(severe_y == 2)[0]).shape[0]
    TN = (np.where(normal_y == 1)[0]).shape[0] + (np.where(normal_o == 1)[0]).shape[0]
    FP = (np.where(normal_y == 2)[0]).shape[0] + (np.where(normal_o == 2)[0]).shape[0]
    FN = (np.where(mtom_y == 1)[0]).shape[0] + (np.where(mtom_o == 1)[0]).shape[0] + (np.where(severe_y == 1)[0]).shape[0] + (np.where(severe_y == 1)[0]).shape[0]
    # print(TP, TN, FP, FN)

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_score = 2 * Precision * Recall / (Precision + Recall)
    print("osa_{} precision: {}".format(model_type, Precision))
    print("osa_{} recall: {}".format(model_type, Recall))
    print("osa_{} f1 score: {}".format(model_type), F1_score)



