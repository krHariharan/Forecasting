import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
from vmdpy import VMD
from parameter_gen import parameter_gen

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, other_params = None):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        if other_params is not None:
            a = np.append(a, other_params.iloc[i:(i+look_back), :])
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

dataframe = pd.read_csv("../data/processed/"+inputFile, index_col=0)
dataframe.index = pd.to_datetime(dataframe.index)

iron_df = pd.read_csv("../data/processed/simple_iron_ore.csv", index_col=0)
iron_df.index = pd.to_datetime(iron_df.index)
iron_df.rename(columns={"Price": "Iron_Price"}, inplace=True)

oil_df = pd.read_csv("../data/processed/simple_crude_oil.csv", index_col=0)
oil_df.index = pd.to_datetime(oil_df.index)
oil_df.rename(columns={"Price": "Oil_Price"}, inplace=True)

silver_df = pd.read_csv("../data/processed/simple_silver.csv", index_col=0)
silver_df.index = pd.to_datetime(silver_df.index)
silver_df.rename(columns={"Price": "Silver_Price"}, inplace=True)

dataframe = pd.merge(dataframe, iron_df, left_index=True, right_index=True)
dataframe = pd.merge(dataframe, oil_df, left_index=True, right_index=True)
dataframe = pd.merge(dataframe, silver_df, left_index=True, right_index=True)

dataframe = dataframe.diff(axis=0).iloc[2:, :]
print(dataframe)
print(dataframe.corr())

param_df = dataframe.iloc[:, 1:]
param_df = (param_df - param_df.min()) / (param_df.max() - param_df.min())
# print(param_df)
dataframe = dataframe["Price"]

datasetFull = dataframe.values
datasetFull = datasetFull.astype('float32').reshape((-1, 1))
print(datasetFull)

#. some sample parameters for VMD  
alpha = 2000       # moderate bandwidth constraint  
tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
K = 20              # 3 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7  


#. Run VMD 
u, u_hat, omega = VMD(datasetFull, alpha, tau, K, DC, init, tol)  

look_back = 3
train_size = int(len(datasetFull) * 0.95)
test_size = len(datasetFull) - train_size
train, test = datasetFull[0:train_size,:], datasetFull[train_size:len(datasetFull),:]
param_train, param_test = param_df.iloc[0:train_size, :], param_df.iloc[train_size:len(datasetFull), :]
# reshape into X=t and Y=t+1
_, trainYfull = create_dataset(train, look_back)
_, testYfull = create_dataset(test, look_back)

trainPredictFull = np.zeros(len(trainYfull))
testPredictFull = np.zeros(len(testYfull))

for i, dataset in enumerate(u):
    print(i)

    dataset = np.reshape(dataset, (-1,1))
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back, param_train)
    testX, testY = create_dataset(test, look_back, param_test)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # for i in range( trainX.shape[0]):
    #     print(trainX[i])
    #     print(trainY[i])
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back*4)))
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
    trainPredictFull += trainPredict[:, 0]
    testPredictFull += testPredict[:, 0]

print(testPredictFull)
print(trainPredictFull)

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
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.savefig("vmd_lstm.png")