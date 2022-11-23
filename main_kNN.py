import statistics
from statistics import stdev
import matplotlib
from matplotlib import pyplot as plt
from sklearn import preprocessing
from kNN import KNeighborsClassifier
matplotlib.use('TkAgg')
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def manage_list(acc, recall, precision, auc):
    metric = []
    label = ''
    for i in range(4):
        if i == 0:
            metric = acc
            label = 'Acurracy'
        if i == 1:
            metric = recall
            label = 'Recall'
        if i == 2:
            metric = precision
            label = 'Precision'
        if i == 3:
            metric = auc
            label = 'AUC'

        output = [metric[i:i + n] for i in range(0, len(metric), n)]

        if i == 0:
            print('-------- Best K / fold ---------')
            for k in k_range:
                val_max = max(output[k - 1])
                pos_max = len(output[k - 1]) - output[k - 1][::-1].index(val_max) - 1
                dic_temp[k] = val_max
                print(k, "valor maximo = ", val_max, "Fold #", pos_max)
            fig, ax = plt.subplots()
            plt.xticks(list(k_range))
            ax.plot(dic_temp.keys(), dic_temp.values())
            ax.set(xlabel="k",
                   ylabel="Accuracy",
                   title=metric_dist)
            plt.show()

        print('---------- Metric - ', label ,' -----------')
        for k in k_range:
            mean = statistics.mean(output[k-1])
            dsv = stdev(output[k-1])
            print('--- k',k,' ----')
            print('Overall ',label, mean)
            print('Standard Deviation is:', dsv)
            print('----------------')


if __name__ == '__main__':

    k_range = range(1, 21)
    scores = []
    recall = []
    precision = []
    auc = []
    dic_temp = {}
    max_value = None
    n = 10
    metric_dist = 'euclidean'
    datos = pd.read_csv('dataset.csv')

    # dataset original
    df = pd.DataFrame(datos, columns=['class', '42', '40', '6', '24', '14', '4', '10', '22'])
    df.to_csv('DATA.csv')

    # dataset normalizado
    scaler = preprocessing.MinMaxScaler()
    values = df.values
    scaledValues = scaler.fit_transform(values)
    df = pd.DataFrame(scaledValues, columns=[0, 42, 40, 6, 24, 14, 4, 10, 22])
    df.to_csv('DATA_normalized.csv')

    # our hipothesys
    X = df[[42, 40, 6, 24, 14, 4, 10, 22]]
    y = df[[0]]

    # set stratified 10-folds CV
    skf = StratifiedKFold(n_splits=n,random_state=1,shuffle=True)

    for k in k_range:
        for train_index, test_index in skf.split(X, y):
            knn = KNeighborsClassifier(k=k, dist_metric=metric_dist)
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

            knn.fit(X_train_fold.values, y_train_fold.values.ravel())
            y_pred = knn.predict(X_test_fold.values)
            pred_prob = knn.predict_proba(X_test_fold.values)

            scores.append(accuracy_score(y_test_fold, y_pred))
            recall.append(recall_score(y_test_fold, y_pred))
            precision.append(precision_score(y_test_fold, y_pred))
            auc.append(roc_auc_score(y_test_fold, pred_prob[:,1]))

    manage_list(scores,recall,precision,auc)

