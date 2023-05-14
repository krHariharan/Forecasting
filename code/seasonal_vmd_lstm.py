# Model decomposing data using seasonal decomposition followed by VMD, and then performing LSTM
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from statsmodels.tsa.seasonal import seasonal_decompose
from vmdpy import VMD
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

# making sure dataset is even sized, so that the output of VMD is also the same size
if datasetFull.shape[0] % 2 == 1:
     datasetFull = datasetFull[1:, :]

# seasonal decomposition
decomposed_dataset = seasonal_decompose(datasetFull, model="multiplicative", period=365, extrapolate_trend='freq')
decomposed_datasets = []

plt.plot(decomposed_dataset.resid)
plt.savefig("resid.png")
plt.clf()
plt.plot(decomposed_dataset.trend)
plt.savefig("trend.png")
plt.clf()
plt.plot(decomposed_dataset.seasonal)
plt.savefig("seasonal.png")
plt.clf()

#. some sample parameters for VMD  
alpha = 2000       # moderate bandwidth constraint  
tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
K = 20              # 20 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7  


#. Run VMD - trend data
u, u_hat, omega = VMD(decomposed_dataset.trend, alpha, tau, K, DC, init, tol)
decomposed_datasets.append(u)

#. Run VMD - residual data
u, u_hat, omega = VMD(decomposed_dataset.resid, alpha, tau, K, DC, init, tol)
decomposed_datasets.append(u)

# split into train and test sets
train_size = int(len(datasetFull) * 0.95)
test_size = len(datasetFull) - train_size
look_back = 10

train, test = datasetFull[0:train_size,:], datasetFull[train_size:len(datasetFull),:]
seasonal_train, seasonal_test = decomposed_dataset.seasonal[0:train_size].reshape(-1,1), decomposed_dataset.seasonal[train_size:len(datasetFull)].reshape(-1,1)
trend_train, trend_test = decomposed_dataset.trend[0:train_size].reshape(-1,1), decomposed_dataset.trend[train_size:len(datasetFull)].reshape(-1,1)
resid_train, resid_test = decomposed_dataset.resid[0:train_size].reshape(-1,1), decomposed_dataset.resid[train_size:len(datasetFull)].reshape(-1,1)
_, trainYfull = create_dataset(train, look_back)
_, testYfull = create_dataset(test, look_back)

_, trainYseasonal = create_dataset(seasonal_train, look_back)
_, testYseasonal = create_dataset(seasonal_test, look_back)

_, trainYtrend = create_dataset(trend_train, look_back)
_, testYtrend = create_dataset(trend_test, look_back)

_, trainYresid = create_dataset(resid_train, look_back)
_, testYresid = create_dataset(resid_test, look_back)

trainPartials = [trainYtrend, trainYresid]
testPartials = [testYtrend, testYresid]


trainPredictFull = trainYseasonal
testPredictFull = testYseasonal

for i, datasets in enumerate(decomposed_datasets):
    trainPredictPartial = np.zeros(len(trainYfull))
    testPredictPartial = np.zeros(len(testYfull))
    for j, dataset in enumerate(datasets):
        print(str(i)+" - "+str(j))
        plt.plot(dataset)
        plt.savefig("seasonal_u_"+str(i)+"_"+str(j)+".png")
        plt.clf()
        # print(dataset)
        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset.reshape(-1, 1))
        # split into train and test sets
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
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
        # calculate root mean squared error - individual VMD components
        trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %f RMSE' % (trainScore))
        testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %f RMSE' % (testScore))
        trainPredictPartial += trainPredict[:, 0]
        testPredictPartial += testPredict[:, 0]
    print(i)
    # calculate root mean squared error - trend/residual component
    trainScore = np.sqrt(mean_squared_error(trainPartials[i], trainPredictPartial))
    print('Train Score: %f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(testPartials[i], testPredictPartial))
    print('Test Score: %f RMSE' % (testScore))
    # add to full prediction
    trainPredictFull *= trainPredictPartial
    testPredictFull *= testPredictPartial

# calculate root mean squared error - full data
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
plt.savefig("seasonal_vmd_lstm.png")