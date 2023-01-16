## Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import sklearn
import matplotlib
import warnings 
import talib

warnings.filterwarnings('ignore')

## READING FILE
df = pd.read_csv('PETR4Daily.csv')

##scaler = StandardScaler()
##df[['Open', 'High', 'Low', 'Close','Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close','Volume']])

## Checking NaNs and Duplicates
print(df.isna().sum())
print('------------------------------')
print(df.count())
print('------------------------------')
print(df.duplicated().value_counts())

## Criando Novas colunas

df['last_day_variation'] = df['Variation']
df['last_day_close'] = df['Close']
df['last_day_opening'] = df['Open']
df['last_day_high'] = df['High']
df['last_day_low'] = df['Low']
df['last_day_vol'] = df['Volume']

df['open_close_diff'] = df['Close'] - df['Open']
df['high_low_diff'] = df['High'] - df['Low']
df['open_close_pct'] = (df['Close'] - df['Open']) / df['Open']
df['high_low_pct'] = (df['High'] - df['Low']) / df['Low']
df['ema_26'] = df['Close'].ewm(span=26).mean()
print(df['ema_26'])
df['ema_12'] = df['Close'].ewm(span=12).mean()
df['ema_diff'] = df['ema_12'] - df['ema_26']
df['macd'] = df['ema_diff'] - df['ema_diff'].ewm(span=9).mean()
""" df['rsi'] = talib.RSI(df['Close'].values, timeperiod=14)

df['willr'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
df['adx'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
df['cci'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
df['atr'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
df['natr'] = talib.NATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
df['adosc'] = talib.ADOSC(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values, fastperiod=3, slow=10)

df['obv'] = talib.OBV(df['Close'].values, df['Volume'].values)
df['trix'] = talib.TRIX(df['Close'].values, timeperiod=30)
df['ultosc'] = talib.ULTOSC(df['High'].values, df['Low'].values, df['Close'].values, timeperiod1=7, timeperiod2=14, timeperiod3=28)
df['dx'] = talib.DX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
df['minus_di'] = talib.MINUS_DI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
df['plus_di'] = talib.PLUS_DI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
df['minus_dm'] = talib.MINUS_DM(df['High'].values, df['Low'].values, timeperiod=14) """

## DEPOIS FAZER UMA FUNCAO
for i in range(1,len(df)):
  df['last_day_variation'][i] = df['Variation'][i-1]
  df['last_day_close'][i] = df['Close'][i-1]
  df['last_day_opening'][i] = df['Open'][i-1]
  df['last_day_high'][i] = df['High'][i-1]
  df['last_day_low'][i] = df['Low'][i-1]
  df['last_day_vol'][i] = df['Volume'][i-1]

  df['open_close_diff'] = df['Close'] - df['Open'][i-1]
  df['high_low_diff'] = df['High'] - df['Low'][i-1]
  df['open_close_pct'] = (df['Close'] - df['Open']) / df['Open'][i-1]
  df['high_low_pct'] = (df['High'] - df['Low']) / df['Low'][i-1]
  df['ema_26'] = df['Close'].ewm(span=26).mean()[i-1]
  df['ema_12'] = df['Close'].ewm(span=12).mean()[i-1]
  df['ema_diff'] = df['ema_12'] - df['ema_26'][i-1]
  df['macd'] = df['ema_diff'] - df['ema_diff'].ewm(span=9).mean()[i-1]
  """ df['rsi'] = talib.RSI(df['Close'].values, timeperiod=14)[i-1]

  df['willr'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)[i-1]
  df['adx'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)[i-1]
  df['cci'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)[i-1]
  df['atr'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)[i-1]
  df['natr'] = talib.NATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)[i-1]
  df['adosc'] = talib.ADOSC(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values, fastperiod=3, slow=10)[i-1]

  df['obv'] = talib.OBV(df['Close'].values, df['Volume'].values)
  df['trix'] = talib.TRIX(df['Close'].values, timeperiod=30)
  df['ultosc'] = talib.ULTOSC(df['High'].values, df['Low'].values, df['Close'].values, timeperiod1=7, timeperiod2=14, timeperiod3=28)[i-1]
  df['dx'] = talib.DX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)[i-1]
  df['minus_di'] = talib.MINUS_DI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)[i-1]
  df['plus_di'] = talib.PLUS_DI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)[i-1]
  df['minus_dm'] = talib.MINUS_DM(df['High'].values, df['Low'].values, timeperiod=14)[i-1] """

## Criando Novas colunas
df['Moving_avg_variation'] = df['Variation']
df['Moving_avg_close'] = df['Close']
df['Moving_avg_opening'] = df['Open']
df['Moving_avg_high'] = df['High']
df['Moving_avg_low'] = df['Low']
df['Moving_avg_vol'] = df['Volume']

features_moving_avg = ['Variation','Open','Close','High','Low','Volume']
new_features = ['Moving_avg_variation','Moving_avg_opening','Moving_avg_close',
                'Moving_avg_high', 'Moving_avg_low','Moving_avg_vol',
                'open_close_diff','high_low_diff','open_close_pct',
                'high_low_pct','ema_26','ema_12','ema_diff','macd',]
""" new_features = ['Moving_avg_variation','Moving_avg_opening','Moving_avg_close',
                'Moving_avg_high', 'Moving_avg_low','Moving_avg_vol',
                'open_close_diff','high_low_diff','open_close_pct',
                'high_low_pct','ema_26','ema_12','ema_diff','macd','rsi',
                'willr','adx','cci','atr','natr','adosc',
                'obv','trix','ultosc','dx','minus_di','plus_di','minus_dm'] """

for j,k in zip(features_moving_avg,new_features):
  variation = 0 
  qtd = 0 
  mean = 0
  for i in range(1,len(df)-1):
    variation += df[j][i-1] 
    qtd += 1
    mean = variation/qtd
    df[k][i] = mean

## Definindo variaveis dependentes e independentes
x = df.drop('Close',axis = 1)
x = x.drop('Date',axis = 1)
x = x.drop('Open',axis = 1)	
x = x.drop('High',axis = 1)	
x = x.drop('Low',axis = 1)	
x = x.drop('Volume',axis = 1)
x = x.drop('Variation',axis = 1)

y = df['Close']

##X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.3)
X_train = x[0:2635] 
print(X_train)
X_test = x[2635:2735]
print(X_test)

Y_train = y[0:2635]
Y_test = y[2735:2834]

from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

dataset_total = pd.DataFrame(X_train)
print(len(dataset_total))
inputs = dataset_total[len(dataset_total) - len(X_test):].values 

y_pred = model.predict(X_test)

#visualising the results
Y_test = list(Y_test)
plt.plot(Y_test, color = 'red', label = 'Real Petrobras Stock Price(petr4)')
plt.plot(y_pred, color = 'blue', label = 'Predicted Petrobras Stock Price(petr4)')
plt.title('Petrobras Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

from sklearn.metrics import r2_score
r2 = r2_score(Y_test, y_pred)
print('r2 score for perfect model is', r2)

Y_test = list(Y_test)
erro = []
for i in range(len(Y_test)):
  erro.append(Y_test[i] - y_pred[i])

erro_medio = sum(erro)/len(erro)