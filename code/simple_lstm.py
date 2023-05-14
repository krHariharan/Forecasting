# Simple LSTM model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
from parameter_gen import parameter_gen

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, other_params = None):
    print(dataset.shape)
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        if other_params is not None:
            a = np.append(a, other_params.iloc[i+look_back, :])
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
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
    
dataframe = read_csv("../data/processed/"+inputFile, index_col=0)
dataframe.index = pd.to_datetime(dataframe.index)

#parameter generation step
params_full_period = parameter_gen(dataframe.index.min(), dataframe.index.max(), (dataframe.index.max() - dataframe.index.min()).days).reset_index(drop=True)
params_10_yrs = parameter_gen(dataframe.index.min(), dataframe.index.max(), 3650).reset_index(drop=True)
params_5_yrs = parameter_gen(dataframe.index.min(), dataframe.index.max(), 1825).reset_index(drop=True)
params_2_yrs = parameter_gen(dataframe.index.min(), dataframe.index.max(), 730).reset_index(drop=True)
annual_params = parameter_gen(dataframe.index.min(), dataframe.index.max(), 365).reset_index(drop=True)
quarterly_params = parameter_gen(dataframe.index.min(), dataframe.index.max(), 121).reset_index(drop=True)
monthly_params = parameter_gen(dataframe.index.min(), dataframe.index.max(), 30).reset_index(drop=True)
weekly_params = parameter_gen(dataframe.index.min(), dataframe.index.max(), 7).reset_index(drop=True)
generated_params = pd.concat([params_full_period, params_10_yrs, params_5_yrs, params_2_yrs, annual_params, quarterly_params, monthly_params, weekly_params], axis=1, join='inner')
# generated_params = params_full_period
print(generated_params.shape)

dataframe = dataframe.reset_index(drop=True)
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.95)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
generated_train, generated_test = generated_params.iloc[0:train_size, :], generated_params.iloc[train_size:len(dataset), :]
# set size of input to LSTM model
look_back = 10
additional_param_count = 0

# find generated parameters with highest correlation (to be given as input to LSTM)
all_params = pd.concat([dataframe, generated_params], axis=1, join='inner')
params_corr = all_params.corr()["Price"].abs()
selected_params = params_corr.nlargest(1+additional_param_count).index
print(params_corr.nlargest(1+additional_param_count))

trainX, trainY = create_dataset(train, look_back, generated_train[selected_params[1:]])
testX, testY = create_dataset(test, look_back, generated_test[selected_params[1:]])
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(testX[0])
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back+additional_param_count)))
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
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.savefig("simple_lstm.png")