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

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100,1000], 'kernel':['linear']},
              {'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_param = grid_search.best_params_

classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 0.4)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


