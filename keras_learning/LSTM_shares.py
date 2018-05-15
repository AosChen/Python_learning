import numpy as np
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def create_dataset(dataset, TIME_SQARE=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-TIME_SQARE-1):
        a = dataset[i:(i+TIME_SQARE), 0]
        dataX.append(a)
        dataY.append(dataset[i + TIME_SQARE, 0])
    return np.array(dataX), np.array(dataY)


datas = read_csv('shares_data.csv',usecols=[2],skiprows=[0])
datas = datas.values
datas.astype('float32')

scaler = MinMaxScaler(feature_range=(0,1))
datas = scaler.fit_transform(datas)

np.random.seed(7)

TIME_SQARE = 10

train_size = int(len(datas) * 0.8)
test_size = len(datas) - train_size
train, test = datas[0:train_size,:], datas[train_size:len(datas),:]

X_train,y_train = create_dataset(train,TIME_SQARE)
X_test,y_test = create_dataset(test,TIME_SQARE)

X_train = np.reshape(X_train,(X_train.shape[0],TIME_SQARE,1))
X_test = np.reshape(X_test,(X_test.shape[0],TIME_SQARE,1))

model = Sequential()
model.add(LSTM(12,input_shape=(TIME_SQARE,1)))
model.add(Dropout(0.2))
model.add(Dense(6))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=2)

trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)

trainPredictPlot = np.empty_like(datas)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[TIME_SQARE:len(trainPredict)+TIME_SQARE, :] = trainPredict

testPredictPlot = np.empty_like(datas)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(TIME_SQARE*2)+1:len(datas)-1, :] = testPredict

datas = scaler.inverse_transform(datas)

plt.plot(datas)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()