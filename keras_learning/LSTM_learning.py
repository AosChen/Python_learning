from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import np_utils
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

def create_dataset(dataset, TIME_SQARE=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-TIME_SQARE-1):
        a = dataset[i:(i+TIME_SQARE), 0]
        dataX.append(a)
        dataY.append(dataset[i + TIME_SQARE, 0])
    return np.array(dataX), np.array(dataY)

dataframe = read_csv('monthly-car-sales-in-quebec-1960.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

np.random.seed(7)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

TIME_SQARE = 3
X_train,y_train = create_dataset(train,TIME_SQARE)
X_test,y_test = create_dataset(test,TIME_SQARE)

X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))

model = Sequential()
model.add(LSTM(4,input_shape=(1,TIME_SQARE)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=2)

trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

trainPredict = scaler.inverse_transform(trainPredict)
y_train = scaler.inverse_transform([y_train])
testPredict = scaler.inverse_transform(testPredict)
y_test = scaler.inverse_transform([y_test])

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[TIME_SQARE:len(trainPredict)+TIME_SQARE, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(TIME_SQARE*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()