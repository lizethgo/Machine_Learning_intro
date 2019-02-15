#Multiclass classification problem
# KNN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('seed_dataset.csv', sep = '\s+|;|:\t]')
X = dataset.iloc[:,0:7].values
Y = dataset.iloc[:,7:8].values
#X = X.astype(np.float)
#X = np.asfarray(X, float)
#Y = np.asfarray(Y, float)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Grid Search
from sklearn.model_selection import GridSearchCV
#parameters = [{'n_neighbors':[5,10,15], 'metric':['minkowski'],'p':[2,3,4]}]

#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           cv = 10,
#                           n_jobs = -1
#                           )

#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_param = grid_search.best_params_

classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = X_train,
                             y = y_train,
                             cv = 10,
                             n_jobs = -1)

mnac = accuracies.mean()
sdac = accuracies.std()
