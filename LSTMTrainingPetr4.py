#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('PETR4Daily.csv')
training_set = dataset.iloc[:,1:2].values
#training_set = dataset.iloc[:,1:].values # pegar todas as colunas

#dollar = pd.read_csv('USDollar')
#crudeOil = pd.read_csv('crudeOil.csv') 
#crudeOil_set = crudeOil.iloc[:,1:3].values


#feature scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc = MinMaxScaler(feature_range = (0,1))
#sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)
#crudeOil_set_scaled = sc.transform(crudeOil_set) 

#creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train = np.array(X_train),np.array(y_train)

#reshaping
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

""" #trainning crude oil
X_train_crudeOil = []
for i in range(60,1258):
    X_train_crudeOil.append(crudeOil_set_scaled[i-60:i,0])
X_train_crudeOil = np.array(X_train_crudeOil)
X_train_crudeOil = np.reshape(X_train_crudeOil,(X_train_crudeOil.shape[0],X_train_crudeOil.shape[1],1))

#concatenate crude oil and petr4 training
X_train = np.concatenate((X_train, X_train_crudeOil), axis=2)
 """
#building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import Bidirectional
from keras.layers import Flatten

regressor = Sequential()

""" #adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
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
regressor.add(Dense(16, activation='relu'))
regressor.add(Dense(8, activation='relu'))
regressor.add(Dense(4, activation='relu'))
regressor.add(Dense(1, activation='sigmoid'))
#regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 100)

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
dataset_total = pd.concat((dataset['Open'],dataset_test['Open']),axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Petrobras Stock Price(petr4)')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Petrobras Stock Price(petr4)')
plt.title('Petrobras Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
 
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