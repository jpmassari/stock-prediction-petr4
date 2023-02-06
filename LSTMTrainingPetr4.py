import sys
import os
from itertools import chain 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import normalize
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical #one_hot
##pip install nvidia-cublas-cu11

#print(tf.config.list_physical_devices('GPU'))
#importing the datase
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dataset = pd.read_csv('PETR4Daily.csv')
training_set = dataset.iloc[2793:2814,1:2].values #dataset.iloc (iloc não usa gpu)

def make_variables(initializer):
    return (tf.Variable(initializer(shape=[19,3], dtype=tf.float64)),
            tf.Variable(initializer(shape=[19,3], dtype=tf.float64)),
            tf.Variable(initializer(shape=[19,3], dtype=tf.float64)))
X_train, y_train, X_test = make_variables(tf.zeros_initializer())

#feature scaling
""" from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc = MinMaxScaler(feature_range = (0,1)) """

training_set = tf.constant(training_set)
training_set_scaled = tf.Variable(training_set)

def difa(a,b):
    diff = (a-b)/b
    if(diff > 0):
        return diff
    else:
        return 0

def normalize(p, n=10, m=100):
    v = p
    nm1 = (n-1)
    v = np.round(v*m) + np.round((n-1)/2)
    if v > nm1: 
        return nm1
    elif v < 0: 
        return 0 
    else: 
        return v

for i in range(0, 20):
    diff_a = difa(training_set[i,0], training_set[i+1,0])
    p = normalize(diff_a)
    #print(int(p))
    training_set_scaled[i].assign(int(p))
print("training_set_scaled: ",training_set_scaled)

for i in range(0, 19):
    for x in range(i, i+1):
        d1 = training_set_scaled[tf.size(y_train) - (tf.size(y_train) - x)]
        d2 = training_set_scaled[x + 1]
        d3 = training_set_scaled[x + 2]
        diff1 = 0
        diff2 = 0
        if(d2 - d1 > 0):
            diff1 = float(str(3) + tf.as_string(training_set_scaled[x]))
        else:
            diff1 = float(str(2) + tf.as_string(training_set_scaled[x]))
        if(d2 - d3 > 0):
            diff2 = float(str(3) + tf.as_string(training_set_scaled[x + 1]))
        else:
            diff2 = float(str(2) + tf.as_string(training_set_scaled[x + 1]))

        y_train[x].assign([200+x, diff1, diff2])
        X_train[x].assign([float(str(1) + tf.as_string(training_set_scaled[x])), float(str(1) + tf.as_string(training_set_scaled[x])), 100+x])

X_train = tf.reshape(X_train, (X_train.shape[0]//1, 1, 3))
y_train = tf.reshape(y_train, (y_train.shape[0]//1, 1, 3))

X_train = tf.cast(X_train, dtype=tf.int32)
y_train = tf.cast(y_train, dtype=tf.int32)
print("y_train: ", y_train)
print("X_train: ", X_train)

#building the RNN
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import CuDNNLSTM
from sklearn import linear_model

regressor = Sequential()

#adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 2024 , return_sequences = True, input_shape = (1,3)))
#regressor.add(Dropout(0.2))
""" #adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 30, return_sequences = True))
#regressor.add(Dropout(0.2))

#adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 30, return_sequences = True))
#regressor.add(Dropout(0.2))

#adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 30, return_sequences = True))
#regressor.add(Dropout(0.2))

#adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 30, return_sequences = True))
#regressor.add(Dropout(0.2))

#adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 30, return_sequences = True))
#regressor.add(Dropout(0.2))
 """
#adding the output layer
regressor.add(Dense(units = 3, activation='softmax')) #sigmoid

#compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

""" X_train = tf.one_hot(X_train, 3)
y_train = tf.one_hot(y_train, 3)
print("to_categorical x_train: ",X_train)
print("to_categorical y_train: ",y_train) """

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/')
regressor.fit(X_train, y_train, epochs = 5000, batch_size=1, callbacks=[tensorboard], verbose=1) #700
print(regressor.metrics_names)
# evaluate the model
scores = regressor.evaluate(X_train, y_train, verbose=0)
print(scores*100)

# Make predictions on the test set
predictions = regressor.predict(X_train)
print("predictions: ", predictions)
regressor.save('lstmPetr4.h5')

# Calculate the accuracy of the model
accuracy = np.sum(predictions == y_train) / len(predictions)
print(accuracy)

s = 0
for i in range(2815,2834):
    for x in range(i, i+1):
        X_test[i-2815].assign([float(str(1) + tf.as_string(training_set_scaled[x])), float(str(1) + tf.as_string(training_set_scaled[x + 1])), 100+s])
        s += 1
X_test = tf.reshape(X_test, (X_test.shape[0]//1, 1, 3))
X_test = tf.cast(X_test, tf.int32)
print("X_test: ",X_test)
predicted_stock_price = regressor.predict(X_test)
#predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print("predicted_stock_price: ",predicted_stock_price)

#getting realstock price to compare prediction
dataset_test = pd.read_csv('PETR4Train.csv')
real_stock_price = dataset_test.iloc[:,1:2].values
real_stock_price = tf.constant(real_stock_price)
print("real stock price: ",real_stock_price)
print(real_stock_price.shape)

#real_stock_price_scaled = sc.transform(real_stock_price)


#visualising the results
plt.plot(real_stock_price, color = 'red')
plt.plot(predicted_stock_price[:,0], color = 'blue')
plt.title('Petrobras Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
 
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(real_stock_price[:19,0], predicted_stock_price)
print(mse)

#calculating the Root mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(real_stock_price[:19,0], predicted_stock_price))
print(rmse)

#calculating the mean_absolute_error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(real_stock_price[:19,0], predicted_stock_price)
print(mae)

#calculating the R2
from sklearn.metrics import r2_score
r2 = r2_score(real_stock_price[:19,0], predicted_stock_price)
print(r2)