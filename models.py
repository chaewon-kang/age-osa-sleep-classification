import numpy as np
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
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
    param_grid = {'hidden_layer_sizes': [(50,), (60,), (50, 50,), (50, 60,)]}
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

