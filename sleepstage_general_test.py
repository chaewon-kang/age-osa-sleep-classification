import pickle
import numpy as np
import models

normal_Y_test = np.array([29, 90, 119, 140])
normal_O_test = np.array([164,165,167,168])
mtom_Y_test = np.array([54, 80,88,107])
mtom_O_test= np.array([16,72,93,103,121,145])
severe_Y_test = np.array([132,175,176,177,179])
severe_O_test = np.array([52,169,170,171,174])
test_num = np.array([normal_O_test, normal_Y_test, mtom_O_test, mtom_Y_test, severe_O_test, severe_Y_test])

general_svm = pickle.load(open("/user/chae2089/CWpaper/FN_REVISION/GENERAL_MODEL/SVM/GENERAL_MODEL_SVM.sav", 'rb'))
models.general_test(general_svm, normal_Y_test)
models.general_test(general_svm, normal_O_test)
models.general_test(general_svm, mtom_Y_test)
models.general_test(general_svm, mtom_O_test)
models.general_test(general_svm, severe_Y_test)
models.general_test(general_svm, severe_O_test)

general_knn = pickle.load(open("/user/chae2089/CWpaper/FN_REVISION/GENERAL_MODEL/kNN/GENERAL_MODEL_kNN.sav", 'rb'))
models.general_test(general_knn, normal_Y_test)
models.general_test(general_knn, normal_O_test)
models.general_test(general_knn, mtom_Y_test)
models.general_test(general_knn, mtom_O_test)
models.general_test(general_knn, severe_Y_test)
models.general_test(general_knn, severe_O_test)

general_mlp = pickle.load(open("/user/chae2089/CWpaper/FN_REVISION/GENERAL_MODEL/MLP/GENERAL_MODEL_MLP.sav", 'rb'))
models.general_test(general_mlp, normal_Y_test)
models.general_test(general_mlp, normal_O_test)
models.general_test(general_mlp, mtom_Y_test)
models.general_test(general_mlp, mtom_O_test)
models.general_test(general_mlp, severe_Y_test)
models.general_test(general_mlp, severe_O_test)






