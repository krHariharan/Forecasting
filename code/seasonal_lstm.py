# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
tf.random.set_seed(7)
# load the dataset
inputFile = ""
if len(sys.argv) > 1:
    inputFile = sys.argv[1]
else:
    print("Enter file to process")
    inputFile = input()
dataframe = read_csv("../data/processed/"+inputFile, usecols=[1])
datasetFull = dataframe.values
datasetFull = datasetFull.astype('float32')

decomposed_dataset = seasonal_decompose(datasetFull, model="multiplicative", period=365, extrapolate_trend='freq')
datasets = [decomposed_dataset.seasonal, decomposed_dataset.trend, decomposed_dataset.resid]

train_size = int(len(datasetFull) * 0.95)
test_size = len(datasetFull) - train_size
look_back = 10

train, test = datasetFull[0:train_size,:], datasetFull[train_size:len(datasetFull),:]
# reshape into X=t and Y=t+1
_, trainYfull = create_dataset(train, look_back)
_, testYfull = create_dataset(test, look_back)

trainPredictFull = np.full(len(trainYfull), 1, 'float32')
testPredictFull = np.full(len(testYfull), 1, 'float32')

for dataset in datasets:
    print(dataset)
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))
    print(dataset)
    # split into train and test sets
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    print(trainX)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %f RMSE' % (testScore))
    # add to full prediction
    trainPredictFull *= trainPredict[:, 0]
    testPredictFull *= testPredict[:, 0]

trainScore = np.sqrt(mean_squared_error(trainYfull, trainPredictFull))
print('Train Score: %f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testYfull, testPredictFull))
print('Test Score: %f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(datasetFull)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredictFull)+look_back, :] = trainPredictFull.reshape(-1,1)
# shift test predictions for plotting
testPredictPlot = np.empty_like(datasetFull)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredictFull)+(look_back*2)+1:len(dataset)-1, :] = testPredictFull.reshape(-1,1)
# plot baseline and predictions
plt.plot(datasetFull)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.savefig("seasonal_lstm.png")