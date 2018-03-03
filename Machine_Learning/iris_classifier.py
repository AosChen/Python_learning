import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

x_train,x_test,y_train,y_test = train_test_split(
    iris_x,iris_y,test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
print(knn.predict(x_test))
print(y_test)