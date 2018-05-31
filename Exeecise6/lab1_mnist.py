import pandas as pd
from BP import BP


data = pd.read_excel('data/数字1~9.xlsx', header=None)
data = data.drop(data.index[[0, 11, 12, 13]], axis=0).drop([0], axis=1).values
data = data.astype(int)

train_data = data[:10]
test_data = data[10:]

x_train, y_train = train_data[:, :-1], train_data[:, -1]
x_test, y_test = test_data[:, :-1], test_data[:, -1]

if __name__ == '__main__':
    model = BP(63, 30, 10, std=0.1)
    history = model.train(x_train, y_train, x_test, y_test, learning_rate=0.05,
                          batch_size=1, num_iters=800, verbose=True)
    print("model_loss: " + str(history['loss_history']))
    print("model_acc: " + str(history['train_acc_history']))
    print("==========================================")
    print("预测： " + str(model.predict(x_test)))
    print("实际： " + str(y_test))
