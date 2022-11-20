import numpy as np
from dataset import class_dataset
from missing_value import class_missing_value
from classifier import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    """
    Memanggil Dataset disease_heart
    """
    dataset = class_dataset(dataset_name="heart")

    """
    missing value
    """
    data = class_missing_value(dataset, missing_name="min")

    "preprocesing (normalisasi)"

    """
    model DTreee
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.17, random_state=42)
    dtree = class_DTree(X_train, y_train)
    dtree.model()
    y_pred = dtree.predict(X_test)

    """
    evaluasi
    """
    acc = accuracy_score(y_test, y_pred)
    print("accuracy : ", acc)



    # """
    # model NB
    # """
    # X_train, X_test, y_train, y_test = train_test_split(
    #     data.X, data.y, test_size=0.17, random_state=42)

    # data.y = np.array([data.y[1] for i in data.y])
    #
    # kf = KFold(n_splits=3, random_state=None, shuffle=False)
    # acc = []
    # for train_index, test_index in kf.split(data.X):
    #     X_train, X_test = data.X[train_index], data.X[test_index]
    #     y_train, y_test = data.y[train_index], data.y[test_index]
    # nb = class_NB(X_train, y_train)
    # nb.model()
    # y_pred = nb.predict(X_test)
    # acc.append(accuracy_score(y_test, y_pred))

    # """
    # model KNN
    # """
    # X_train, X_test, y_train, y_test = train_test_split(
    #     data.X, data.y, test_size=0.2, random_state=42)
    # knn = class_KNN(X_train, y_train)
    # knn.model()
    # y_pred = knn.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print("accuracy : ", acc)

    # data = class_dataset(dataset_name="2D")
    # knn = class_KNN(data.X, data.y)
    # knn.viewDataset()
    # knn.model()
    # data_testing = [[2, 5],[2,2]]
    # knn.predict(data_testing)




