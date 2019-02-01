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

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
