import numpy as np
import pandas as pd

dataset = pd.read_csv('iris.csv')
X = dataset.iloc[:,0:4].values
Y = dataset.iloc[:,-1].values

#Label data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
Y = label_encoder_X.fit_transform(Y)

#in case we need one hot encoder
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()

#spliting data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state = 1)

#Standarizing data
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)

#SVM linear
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
accuracy_2 = classifier.score(X_test,Y_test)

from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(Y_test, y_pred)
print(cm_1)

#SVM Gaussian
from sklearn.svm import SVC
classifier_g = SVC(kernel = 'rbf', random_state = 0)
classifier_g.fit(X_train, Y_train)
y_pred = classifier_g.predict(X_test)
accuracy_2 = classifier_g.score(X_test,Y_test)

cm_2 = confusion_matrix(Y_test, y_pred)
print(cm_2)

#K-NN
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_KNN.fit(X_train, Y_train)
y_pred_KNN = classifier_g.predict(X_test)
accuracy_KNN = classifier_g.score(X_test,Y_test)

cm_KNN = confusion_matrix(Y_test, y_pred)
print(cm_KNN)


