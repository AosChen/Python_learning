import pandas as pd
import numpy as np
from .BP import BP

iris = pd.read_csv('data/iris.txt', header=None)
iris[4] = iris[4].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
iris = iris.values

np.random.shuffle(iris)

x_train = iris[int(iris.shape[0] * 0.1):, :-1]
y_train = iris[int(iris.shape[0] * 0.1):, -1].astype(int)
x_test = iris[:int(iris.shape[0] * 0.1), :-1]
y_test = iris[:int(iris.shape[0] * 0.1), -1].astype(int)

if __name__ == '__main__':
    model = BP(4, 10, 3, std=0.1)
    history = model.train(x_train, y_train, x_test, y_test, learning_rate=0.03,
                          batch_size=128, num_iters=800, verbose=True)
    print("model_loss: " + str(history['loss_history']))
    print("model_acc: " + str(history['val_acc_history']))
    print("==========================================")
    print("预测： " + str(model.predict(x_test)))
    print("实际： " + str(y_test))
