import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
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
            a = np.append(a, other_params.iloc[i+look_back, :])
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

params_10_yrs = parameter_gen(dataframe.index.min(), dataframe.index.max(), 3650).reset_index(drop=True)
params_5_yrs = parameter_gen(dataframe.index.min(), dataframe.index.max(), 3650).reset_index(drop=True)
params_2_yrs = parameter_gen(dataframe.index.min(), dataframe.index.max(), 3650).reset_index(drop=True)
annual_params = parameter_gen(dataframe.index.min(), dataframe.index.max(), 365).reset_index(drop=True)
quarterly_params = parameter_gen(dataframe.index.min(), dataframe.index.max(), 121).reset_index(drop=True)
monthly_params = parameter_gen(dataframe.index.min(), dataframe.index.max(), 30).reset_index(drop=True)
weekly_params = parameter_gen(dataframe.index.min(), dataframe.index.max(), 7).reset_index(drop=True)
generated_params = pd.concat([annual_params, quarterly_params, monthly_params, weekly_params], axis=1, join='inner')

datasetFull = dataframe.values
datasetFull = datasetFull.astype('float32')

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
train_size = int(len(datasetFull) * 0.67)
test_size = len(datasetFull) - train_size
train, test = datasetFull[0:train_size,:], datasetFull[train_size:len(datasetFull),:]
# reshape into X=t and Y=t+1
_, trainYfull = create_dataset(train, look_back)
_, testYfull = create_dataset(test, look_back)

trainPredictFull = np.zeros(len(trainYfull))
testPredictFull = np.zeros(len(testYfull))

for dataset in u:
    dataset = np.reshape(dataset, (-1,1))
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
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
plt.savefig("simple_lstm.png")