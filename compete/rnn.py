# -*- coding: UTF-8 -*-

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, classification_report
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Masking
from sklearn import preprocessing
from time import time

# 共238个特征
feature_size = 238
# 每批样本数量
batch_size = 256
# 共8类
num_classes = 8
# 共N次迭代
epochs = 20

DIR = 'D:\\研究所\\比赛\\features\\'



def data_prepare():

    FILE_NAME = []
    for i in range(num_classes):
        FILE_NAME.append('original_data_Label' + str(i + 1) + '_features.txt')

    data1 = np.loadtxt(DIR+FILE_NAME[0], dtype=float)
    dataset1 = np.random.permutation(data1)
    print(dataset1.shape)

    data2 = np.loadtxt(DIR+FILE_NAME[1], dtype=float)
    dataset2 = np.random.permutation(data2)
    print(dataset2.shape)

    data3 = np.loadtxt(DIR+FILE_NAME[2], dtype=float)
    dataset3 = np.random.permutation(data3)
    print(dataset3.shape)

    data4 = np.loadtxt(DIR+FILE_NAME[3], dtype=float)
    dataset4 = np.random.permutation(data4)
    print(dataset4.shape)

    data5 = np.loadtxt(DIR+FILE_NAME[4], dtype=float)
    dataset5 = np.random.permutation(data5)
    print(dataset5.shape)

    data6 = np.loadtxt(DIR+FILE_NAME[5], dtype=float)
    dataset6 = np.random.permutation(data6)
    print(dataset6.shape)

    data7 = np.loadtxt(DIR+FILE_NAME[6], dtype=float)
    dataset7 = np.random.permutation(data7)
    print(dataset7.shape)

    data8 = np.loadtxt(DIR+FILE_NAME[7], dtype=float)
    dataset8 = np.random.permutation(data8)
    print(dataset8.shape)

    len1 = len(dataset1)
    len2 = len(dataset2)
    len3 = len(dataset3)
    len4 = len(dataset4)
    len5 = len(dataset5)
    len6 = len(dataset6)
    len7 = len(dataset7)
    len8 = len(dataset8)

    min_len = min(len1, len2, len3, len4, len5, len6, len7, len8)  # 取数据量最少的那一类数据的80%作为训练样本
    # The index
    k = int(np.ceil(min_len * 0.8))

    train_set1 = dataset1[0:k, :]
    train_set2 = dataset2[0:k, :]
    train_set3 = dataset3[0:k, :]
    train_set4 = dataset4[0:k, :]
    train_set5 = dataset5[0:k, :]
    train_set6 = dataset6[0:k, :]
    train_set7 = dataset7[0:k, :]
    train_set8 = dataset8[0:k, :]

    test_set1 = dataset1[k:len1, :]
    test_set2 = dataset2[k:len2, :]
    test_set3 = dataset3[k:len3, :]
    test_set4 = dataset4[k:len4, :]
    test_set5 = dataset5[k:len5, :]
    test_set6 = dataset6[k:len6, :]
    test_set7 = dataset7[k:len7, :]
    test_set8 = dataset8[k:len8, :]

    # 标准化 z-score
    scaler1 = preprocessing.StandardScaler().fit(train_set1)
    scaler2 = preprocessing.StandardScaler().fit(train_set2)
    scaler3 = preprocessing.StandardScaler().fit(train_set3)
    scaler4 = preprocessing.StandardScaler().fit(train_set4)
    scaler5 = preprocessing.StandardScaler().fit(train_set5)
    scaler6 = preprocessing.StandardScaler().fit(train_set6)
    scaler7 = preprocessing.StandardScaler().fit(train_set7)
    scaler8 = preprocessing.StandardScaler().fit(train_set8)

    train_set1 = scaler1.transform(train_set1)
    train_set2 = scaler2.transform(train_set2)
    train_set3 = scaler3.transform(train_set3)
    train_set4 = scaler4.transform(train_set4)
    train_set5 = scaler5.transform(train_set5)
    train_set6 = scaler6.transform(train_set6)
    train_set7 = scaler7.transform(train_set7)
    train_set8 = scaler8.transform(train_set8)

    # print(train_set3.mean(axis=0))
    # print(train_set3.std(axis=0))
    # plt.scatter(np.arange(train_set3.shape[0]), train_set3[:, 8])
    # plt.show()

    test_set1 = scaler1.transform(test_set1)
    test_set2 = scaler2.transform(test_set2)
    test_set3 = scaler3.transform(test_set3)
    test_set4 = scaler4.transform(test_set4)
    test_set5 = scaler5.transform(test_set5)
    test_set6 = scaler6.transform(test_set6)
    test_set7 = scaler7.transform(test_set7)
    test_set8 = scaler8.transform(test_set8)

    # 贴标签
    label1 = 1 * np.ones((len(train_set1), 1))
    train_set1 = np.concatenate((train_set1, label1), axis=1)
    label1 = 1 * np.ones((len(test_set1), 1))
    test_set1 = np.concatenate((test_set1, label1), axis=1)

    label2 = 2 * np.ones((len(train_set2), 1))
    train_set2 = np.concatenate((train_set2, label2), axis=1)
    label2 = 2 * np.ones((len(test_set2), 1))
    test_set2 = np.concatenate((test_set2, label2), axis=1)

    label3 = 3 * np.ones((len(train_set3), 1))
    train_set3 = np.concatenate((train_set3, label3), axis=1)
    label3 = 3 * np.ones((len(test_set3), 1))
    test_set3 = np.concatenate((test_set3, label3), axis=1)

    label4 = 4 * np.ones((len(train_set4), 1))
    print(train_set4.shape)
    print(label4.shape)
    train_set4 = np.concatenate((train_set4, label4), axis=1)


    label4 = 4 * np.ones((len(test_set4), 1))
    test_set4 = np.concatenate((test_set4, label4), axis=1)

    train_data = np.concatenate((train_set1, train_set2, train_set3, train_set4), axis=0)
    test_data = np.concatenate((test_set1, test_set2, test_set3, test_set4), axis=0)

    return train_data, test_data


# 数据处理的关键方法
def load_dataForLSTM(training_data, test_data):
    X_train = training_data[:, 0:feature_size]  # sample 169
    Y_train = training_data[:, feature_size]    # label
    X_test = test_data[:, 0:feature_size]
    Y_test = test_data[:, feature_size]

    # 根据时间步进行的预处理操作
    time_step = 5
    X_train_list = X_train.tolist()
    X_test_list = X_test.tolist()
    temp_train = list()
    temp_test = list()
    for i in range(0, len(X_train) - time_step):
        temp_train.extend(X_train_list[i: i + time_step])
    for i in range(0, len(X_test) - time_step):
        temp_test.extend(X_test_list[i: i + time_step])

    X_train_lstm = np.array(temp_train)
    X_test_lstm = np.array(temp_test)

    X_train_lstm = X_train_lstm.reshape(len(X_train) - time_step, feature_size * time_step)
    X_test_lstm = X_test_lstm.reshape(len(X_test) - time_step, feature_size * time_step)

    print(X_train_lstm.shape)

    # 更改样本形状
    x_train = X_train_lstm.reshape(X_train_lstm.shape[0], 5, 169)
    x_test = X_test_lstm.reshape(X_test_lstm.shape[0], 5, 169)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    Y_trainx = Y_train[time_step: len(Y_train)]
    Y_testx = Y_test[time_step: len(Y_test)]

    # 类别标签编码
    encoder = LabelEncoder()
    encoded_TrainY = encoder.fit_transform(Y_trainx)
    encoded_TestY = encoder.fit_transform(Y_testx)
    y_train = np_utils.to_categorical(encoded_TrainY, num_classes)
    y_test = np_utils.to_categorical(encoded_TestY, num_classes)

    return x_train, x_test, y_train, y_test, Y_testx


# 搭建LSTM网络结构
def build_LSTM_model():
    model = Sequential()

    # 输出维度50
    model.add(Masking(mask_value=0., input_shape=(5, 169)))
    model.add(LSTM(64,  activation='tanh', kernel_initializer='glorot_uniform', dropout=0.2,
                   recurrent_dropout=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 搭建GRU网络结构
def build_GRU_model():
    model = Sequential()

    # 输出维度
    model.add(GRU(64, activation='tanh', kernel_initializer='glorot_uniform', input_shape=(5, 169), dropout=0.2,
                  recurrent_dropout=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 自定义预测函数（可以不使用）
def predict(model, x_test, y_test):
    predictions = model.predict(x_test)
    get_class = lambda classes_probabilities: np.argmax(classes_probabilities) + 1
    y_pred = np.array(map(get_class, predictions))
    if y_test is not None:
        y_true = np.array(map(get_class, y_test))
        print("准确率：" + str(accuracy_score(y_true, y_pred)))
    return y_true, y_pred


# 自定义混淆矩阵打印
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# 运行模型
def run_model(training_data, test_data):
    # 加载训练测试数据
    x_train, x_test, y_train, y_test, Y_test = load_dataForLSTM(training_data, test_data)
    print("看shapes啦")
    print(x_train.shape)
    print(y_train.shape)

    # 开始LSTM训练
    start = time()
    model = build_LSTM_model()
    model.summary()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_split=0.2)
    model.save('LSTM.h5')
    stop = time()
    print(str(stop - start) + "秒 LSTM训练")
    scores = model.evaluate(x_test, y_test)
    print("=========")
    print(scores)
    print("=========")

    # 开始GRU训练
    start0 = time()
    model0 = build_GRU_model()
    history0 = model0.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                          validation_split=0.2)
    model0.save('GRU.h5')
    stop0 = time()
    print(str(stop0 - start0) + "秒 GRU训练")
    scores0 = model0.evaluate(x_test, y_test)
    print("=========")
    print(scores0)
    print("=========")

    # summarize history for validation
    plt.figure()
    plt.grid(True, linewidth=0.4)
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['val_loss'], 'g--')
    plt.title('LSTM: acc & loss')
    plt.ylabel('val_acc')
    plt.xlabel('Epoch')
    plt.legend(['acc', 'loss'], loc='center right')
    plt.show()

    # 预测：y_true & y_pred
    s = time()
    y_true, y_pred = predict(model, x_test, y_test)
    e = time()
    print(str(e - s) + "秒 测试")
    y_true0, y_pred0 = predict(model0, x_test, y_test)
    report = classification_report(y_true0, y_pred0)
    print('classify_report:')
    print(report)

    # 重新编码标签 为后续绘制ROC曲线
    encoder = LabelEncoder()
    encoded_true = encoder.fit_transform(y_true)
    encoded_pred = encoder.fit_transform(y_pred)
    encoded_true0 = encoder.fit_transform(y_true0)
    encoded_pred0 = encoder.fit_transform(y_pred0)
    roc_true = np_utils.to_categorical(encoded_true, num_classes)
    roc_pred = np_utils.to_categorical(encoded_pred, num_classes)
    roc_true0 = np_utils.to_categorical(encoded_true0, num_classes)
    roc_pred0 = np_utils.to_categorical(encoded_pred0, num_classes)

    # 绘制ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(roc_true.ravel(), roc_pred.ravel())
    fpr["micro0"], tpr["micro0"], _ = roc_curve(roc_true0.ravel(), roc_pred0.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc["micro0"] = auc(fpr["micro0"], tpr["micro0"])
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='LSTM ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='cornflowerblue', linewidth=2)
    plt.plot(fpr["micro0"], tpr["micro0"],
             label='GRU ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro0"]),
             color='darkorange', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC of LSTM & GRU')
    plt.legend(loc="lower right")
    plt.show()

    # 打印混淆矩阵
    cnf_matrix = confusion_matrix(Y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    class_names = ['bus', 'car', 'train', 'metro']
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
    plt.show()


def main():

    # 导入数据
    training_data, test_data = data_prepare()

    # 模型对应的预测结果（列表形式）
    run_model(training_data, test_data)


if __name__ == '__main__':
    main()
