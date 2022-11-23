import numpy as np


class KNeighborsClassifier:

    def __init__(self, k, dist_metric):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _predict(self, X_test):
        neighbors = []
        for x in X_test:
            if self.dist_metric == 'euclidean':
                distances = self.euclidean(x, self.X_train)
                y_sorted = []
                for y in sorted(zip(distances, self.y_train)):
                    y_sorted.append(y[1])
                # append 10 min distances for current point x.
                neighbors.append(y_sorted[:self.k])

            elif self.dist_metric == 'manhattan':
                distances = self.manhattan(x, self.X_train)
                y_sorted = []
                for y in sorted(zip(distances, self.y_train)):
                    y_sorted.append(y[1])
                neighbors.append(y_sorted[:self.k])

        return list(map(self.most_common, neighbors))

    def predict(self, X_test):
        y_pred = self._predict(X_test)
        return y_pred

    def most_common(self, lst):
        return max(set(lst), key=lst.count)

    def euclidean(self, point, data):
        return np.sqrt(np.sum((point - data) ** 2, axis=1))

    def manhattan(self, point, data):
        return np.sum(abs(point - data), axis=1)

    def predict_proba(self, X_test):
        neighbors = []
        for x in X_test:
            if self.dist_metric == 'euclidean':
                distances = self.euclidean(x, self.X_train)
                y_sorted = []
                for y in sorted(zip(distances, self.y_train)):
                    y_sorted.append(y[1])
                neighbors.append(y_sorted[:self.k])

            elif self.dist_metric == 'manhattan':
                distances = self.manhattan(x, self.X_train)
                y_sorted = []
                for y in sorted(zip(distances, self.y_train)):
                    y_sorted.append(y[1])
                neighbors.append(y_sorted[:self.k])
        pred = []
        for n in neighbors:
            ne = np.array(n)

            count0 = (ne == 0).sum() / len(ne)
            count1 = (ne == 1).sum() / len(ne)
            pred.append((count0, count1))

        return (np.array(pred))

