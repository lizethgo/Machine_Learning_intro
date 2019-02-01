#K-NN algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('pima-indians-diabetes.data.csv')
X = dataset.iloc[:,0:8].values
Y = dataset.iloc[:,-1].values
#X = np.asfarray(X, float)
#Y = Y.astype(np.float)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier as KNC
classifier = KNC(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

from sklearn.model_selection import GridSearchCV
parameters = [{'n_neighbors':[5,6,7,8,9,10,20,50], 'metric':['minkowski'], 'p':[2]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_param = grid_search.best_params_

classifier = KNC(n_neighbors = 9, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_new = confusion_matrix(Y_test, y_pred)


