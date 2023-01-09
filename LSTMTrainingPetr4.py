#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#importing the dataset
dataset = pd.read_csv('PETR4Daily.csv')
training_set = dataset.iloc[:,1:2].values
#training_set = dataset.iloc[:,1:].values # pegar todas as colunas

crudeOil = pd.read_csv('crudeOil.csv') 
crudeOil_set = crudeOil.iloc[:,2:3].values

dollar = pd.read_csv('USDollar.csv')
dollar_set = dollar.iloc[:,2:3].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc = MinMaxScaler(feature_range = (0,1))
#sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)
crudeOil_set_scaled = sc.transform(crudeOil_set) 
dollar_set_scaled = sc.transform(dollar_set)


#creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(230, 2761): #60, 1258
    X_train.append(training_set_scaled[i-230:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train = np.array(X_train),np.array(y_train)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

X_train_crude = []
for i in range(230,2761):
    X_train_crude.append(crudeOil_set_scaled[i-230:i,0])
X_train_crude = np.array(X_train_crude)
X_train_crude = np.reshape(X_train_crude,(X_train_crude.shape[0],X_train_crude.shape[1],1))

X_train_dollar = []
for i in range(230,2761):
    X_train_dollar.append(dollar_set_scaled[i-230:i,0])
X_train_dollar = np.array(X_train_dollar)
X_train_dollar = np.reshape(X_train_dollar,(X_train_dollar.shape[0],X_train_dollar.shape[1],1))

X_train = np.concatenate((X_train, X_train_crude, X_train_dollar), axis=2)

#building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import CuDNNLSTM

regressor = Sequential()
""" 
#adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],X_train.shape[2])))
regressor.add(Dropout(0.2))

#adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#adding the output layer
regressor.add(Dense(units = 1))

#compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') """

import random

random_seed = 42
random.seed(random_seed)

regressor.add(Bidirectional(LSTM(units=256, input_shape=(X_train.shape[1], 1)))) #com 124 foi excelente.
regressor.add(Dropout(0.2, noise_shape=None, seed=random_seed))
regressor.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
regressor.add(Dropout(0.2))
regressor.add(Dense(128, activation='relu'))
regressor.add(Dropout(0.2))
regressor.add(Dense(64, activation='relu'))
regressor.add(Dropout(0.2))
regressor.add(Dense(32, activation='relu'))
regressor.add(Dropout(0.2))
regressor.add(Dense(16, activation='relu'))
regressor.add(Dense(8, activation='relu'))
regressor.add(Dense(4, activation='relu'))
regressor.add(Dense(1, activation='sigmoid'))
#regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/')
regressor.fit(X_train, y_train, epochs = 100, callbacks=[tensorboard])

# evaluate the model
scores = regressor.evaluate(X_train, y_train, verbose=0)
print(scores*100)

# Make predictions on the test set
predictions = regressor.predict(X_train)
regressor.save('lstmPetr4.h5')

# Calculate the accuracy of the model
accuracy = np.sum(predictions == y_train) / len(predictions)
print(accuracy)

#making the predictions and visualising the results
dataset_test = pd.read_csv('PETR4Train.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

#getting the predicted stock price of 2017
#dataset_total = pd.concat((dataset['Open'],dataset_test['Open']),axis = 0)
#dataset_total_crude = pd.concat((crudeOil['Open'],dataset_test['Open']),axis = 0)
#dataset_total_dollar = pd.concat((dollar['Open'],dataset_test['Open']),axis = 0)

dataset_total = pd.DataFrame(dataset['Open'])
dataset_total_crude = pd.DataFrame(crudeOil['Open'])
dataset_total_dollar = pd.DataFrame(dollar['Open'])

#inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:].values #21.31 ... 22.87 ( len(dataset_test) 20 - 60 = 80 valores para treino)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 230:].values 
print(len(inputs))
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


inputs_crude = dataset_total_crude[len(dataset_total_crude) - len(dataset_test) - 230:].values
inputs_crude = inputs_crude.reshape(-1,1)
inputs_crude = sc.transform(inputs_crude)

inputs_dollar = dataset_total_dollar[len(dataset_total_dollar) - len(dataset_test) - 230:].values
inputs_dollar = inputs_dollar.reshape(-1,1)
inputs_dollar = sc.transform(inputs_dollar)

X_test = []
for i in range(230,250): #80 precisa bater com a quantia de valores de treino no inputs 
    X_test.append(inputs[i-230:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

X_test_crude = []
for i in range(230,250):
    X_test_crude.append(inputs_crude[i-230:i,0])
X_test_crude = np.array(X_test_crude)
X_test_crude = np.reshape(X_test_crude,(X_test_crude.shape[0],X_test_crude.shape[1],1))

X_test_dollar = []
for i in range(230,250):
    X_test_dollar.append(inputs_dollar[i-230:i,0])
X_test_dollar = np.array(X_test_dollar)
X_test_dollar = np.reshape(X_test_dollar,(X_test_dollar.shape[0],X_test_dollar.shape[1],1))

X_test = np.concatenate((X_test, X_test_crude, X_test_dollar), axis=2)

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualising the results
plt.plot(real_stock_price, color = 'red')
plt.plot(predicted_stock_price, color = 'blue')
plt.title('Petrobras Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
 
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(real_stock_price, predicted_stock_price)
print(mse)

#calculating the Root mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(rmse)

#calculating the mean_absolute_error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(real_stock_price, predicted_stock_price)
print(mae)

#calculating the R2
from sklearn.metrics import r2_score
r2 = r2_score(real_stock_price, predicted_stock_price)
print(r2)

""" import sqlite3  DON'T FORGET TO INSERT THE REGRESSION AS WELL WITH THE AVERAGE

# connect to database
conn = sqlite3.connect('stock_db.db')

# Calculate the average of the predicted stock price and the real stock price
average_price = (predicted_stock_price + real_stock_price)/2

# Insert the average price into the database
cursor = conn.cursor()
cursor.execute("CREATE TABLE stock_prices (average_price REAL, regression_data TEXT, predicted_stock_price REAL)")
cursor.execute("INSERT INTO stock_prices (average_price, regression_data, predicted_stock_price) VALUES (?,?,?)", (average_price, regression_data, predicted_stock_price))

# Commit the changes to the database
conn.commit()

# Close the connection to the database
conn.close() """