#Random Forest Tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('sonar.csv')
X = dataset.iloc[:,0:60].values
Y = dataset.iloc[:,[-1]].values
#X = np.asfarray(X, float)
#Y = Y.astype(np.float)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier as RFC
classifier = RFC(n_estimators = 5, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[50,40,30,20], 'criterion':['entropy']}]
 
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_param = grid_search.best_params_

classifier = RFC(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
cm_new = confusion_matrix(Y_test, y_pred)

#applying cross validation score
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
mean = accuracies.mean()
std = accuracies.std()
