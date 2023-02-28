import sys
import os
import itertools
import tensorflow as tf
import pandas as pd
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

import callback as MyCallback

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#from keras.utils import normalize
#from keras.preprocessing.text import one_hot
#from keras.utils import to_categorical #one_hot

#print(tf.config.list_physical_devices('GPU'))
#importing the datase
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dataset = pd.read_csv('PETR4Daily.csv')
training_set = dataset.iloc[:2815,1:2].values #dataset.iloc (iloc não usa gpu)
training_set_petr4_volume = dataset.iloc[:2814,5:6].values

""" dataset_dollar = pd.read_csv('USDollar.csv')
training_set_dollar = dataset_dollar.iloc[:2814,2:3].values 

dataset_oil = pd.read_csv('crudeOil.csv')
training_set_oil = dataset_oil.iloc[:2814,2:3].values  """

def make_variables(initializer):
    return (tf.Variable(initializer(shape=[2812,4], dtype=tf.float64)),
            tf.Variable(initializer(shape=[2812,3], dtype=tf.float64)),
            tf.Variable(initializer(shape=[2812,4], dtype=tf.float64)),
            tf.Variable(initializer(shape=[2815,1], dtype=tf.int32)))
X_train, y_train, X_test, training_set_scaled = make_variables(tf.zeros_initializer())

#feature scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc = MinMaxScaler(feature_range = (0,1))

training_set = tf.constant(training_set)
#training_set_scaled = tf.Variable(tf.fill(2815),0)

""" training_set_dollar = tf.constant(training_set_dollar)
training_set_scaled_dollar = tf.Variable(training_set_dollar, dtype=tf.float64)

training_set_oil = tf.constant(training_set_oil)
training_set_scaled_oil = tf.Variable(training_set_oil, dtype=tf.float64) """

def subset_pairs(tensor, subset_size):
    def pairwise_combinations(subset):
        pairs = tf.stack(tf.meshgrid(*[subset for _ in range(subset_size)]), axis=-1)
        pairs = tf.reshape(pairs, (-1, subset_size))
        return pairs

    num_elements = tensor.get_shape().as_list()[0]
    indices = tf.range(num_elements)
    combinations = tf.data.Dataset.from_tensor_slices(indices).window(subset_size, shift=1, drop_remainder=True)
    combinations = combinations.flat_map(lambda window: tf.data.Dataset.zip((window.batch(subset_size),)))
    pairs = combinations.map(pairwise_combinations).unbatch()
    return pairs

""" tensor = tf.Variable(tf.range(12))
subset_size = 5
subsets = subset_pairs(tensor, subset_size)
for subset in subsets:
    print("subsets: ",subset.numpy()) """

def cv(x):
    return (tf.math.reduce_std(x)/tf.math.reduce_mean(x))*100

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

for i in range(0, 2814):
    diff_a = difa(training_set[i,0], training_set[i+1,0])
    p = normalize(diff_a, 10, 800)

    """diff_a_dollar = difa(training_set_dollar[i,0], training_set_dollar[i+1,0])
    p_dollar = normalize(diff_a_dollar, 10, 800)

    diff_a_oil = difa(training_set_oil[i,0], training_set_oil[i+1,0])
    p_oil= normalize(diff_a_oil, 10, 800) """
    if(training_set[i,0] < training_set[i+1,0]):
        training_set_scaled[i].assign(int(float((str(3)+tf.strings.as_string(p)))))
    else:
        training_set_scaled[i].assign(int(float((str(2)+tf.strings.as_string(p)))))
    """training_set_scaled_dollar[i].assign(int(p_dollar))
    training_set_scaled_oil[i].assign(int(p_oil)) """
print("training_set_scaled: ",training_set_scaled)

training_set_scaled = tf.cast(training_set_scaled, dtype=tf.int32)
def sliding_windows(input_tensor, window_size):
    num_windows = tf.shape(input_tensor)[0] - window_size + 1
    indices = tf.range(num_windows)[:, None] + tf.range(window_size)[None, :]
    mapped = tf.map_fn(lambda x: tf.gather(input_tensor, x), indices, dtype=tf.int32)
    return tf.squeeze(mapped)
slind = sliding_windows(training_set_scaled, 3)
print("slind: ", slind)

v = []
x = []
a = 0
for row in enumerate(slind):
    print(row)
    subsets = subset_pairs(tf.constant([row]), 3)
    for subset in subsets:
        #print(subset.numpy())
        #X_train[i].assign(subset)
        v.append(subset.numpy())
        print(subset)
        a +=a
    x.append(v)
    print("X: ", x)
print("FINISHED X: ",x)
print("v shape: ",v.shape)
print("x shape: ",x.shape)
print(tf.convert_to_tensor(x))
cp.histogram()
for i in range(0, 2812):
    for x in range(i, i+1):
        d1 = training_set_scaled[tf.size(y_train) - (tf.size(y_train) - x)]
        d2 = training_set_scaled[x + 1]
        d3 = training_set_scaled[x + 2]
        v = cv([d1,d2])
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

        #if x % 2 == 0: #even
        y_train[x].assign([200+x, diff1, diff2])
        X_train[x].assign([
            float(str(1) + tf.as_string(training_set_scaled[x])),
            float(str(1) + tf.as_string(training_set_scaled[x + 1])), 
            #float(training_set_scaled_dollar[x]), 
            #float(training_set_scaled_dollar[x + 1]), 
            #float(training_set_oil[x]), 
            #float(training_set_oil[x + 1]), 
            float(v),
            100+x
            ])
        #else: #odd
            #y_train[x].assign([200+x, diff2, diff1])
            #X_train[x].assign([float(v), float(str(1) + tf.as_string(training_set_scaled[x])), float(training_set_scaled_dollar[x + 1]), float(training_set_scaled_dollar[x]), float(str(1) + tf.as_string(training_set_scaled[x+1])), 100+x]) """
        

X_train = tf.reshape(X_train, (X_train.shape[0]//1, 1, 4))
y_train = tf.reshape(y_train, (y_train.shape[0]//1, 1, 3))
print("y_train: ", y_train)
print("X_train: ", X_train)

X_train = tf.cast(X_train, dtype=tf.int32)
y_train = tf.cast(y_train, dtype=tf.int32)


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
regressor.add(LSTM(units = 8192, return_sequences = True, input_shape = (1,8)))
#regressor.add(Dropout(0.2))

#adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 128, return_sequences = True))
#regressor.add(Dropout(0.2))

#adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 128, return_sequences = True))
#regressor.add(Dropout(0.2))

#adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 128, return_sequences = True))
#regressor.add(Dropout(0.2))

""" #adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 30, return_sequences = True))
#regressor.add(Dropout(0.2))
 """
#adding the output layer
regressor.add(Dense(units = 3, activation='logSoftmax')) #sigmoid

#compiling the RNN
regressor.compile(optimizer = tf.optimizers.experimental.SGD(0.01, 0.9, nesterov=True), loss = tf.metrics.categorical_crossentropy, metrics = tf.metrics.categorical_accuracy )
#categorical_crossentropy
""" X_train = tf.one_hot(X_train, 3)
y_train = tf.one_hot(y_train, 3)
print("to_categorical x_train: ",X_train)
print("to_categorical y_train: ",y_train) """

my_callback = MyCallback(regressor)
regressor.fit(X_train, y_train, epochs = 50, batch_size=9, callbacks=[my_callback], verbose=1) #batch_size 9 -> 9/16 = 192 record
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