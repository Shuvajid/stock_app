import pandas as pd
import numpy as np
from datetime import datetime as dt
np.random.seed(1000)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
from scipy import stats
from scipy.stats import mannwhitneyu 
import pickle
import streamlit as st
import yfinance as yf
import datetime



st.subheader('Query parameters')
start_date = st.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.date_input("End date", datetime.date(2021, 1, 31))

# Retrieving tickers data
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
tickerSymbol = st.selectbox('Stock ticker', ticker_list) # Select ticker symbol
tickerData = yf.Ticker(tickerSymbol) # Get ticker data
dataset = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker






st.subheader('STOCK PEICE LIST')
# if your dataset contains missing value, check which column has missing values
dataset.isnull().sum()





StockPrice = dataset['Close']

st.write(dataset.tail(10))

## Slide Show
historical_days = []
this_day = []
step = 1
max_len = 5 ### lagged variable/ Sliding Window
for idx in range(0, len(StockPrice) - max_len, step):
    historical_days.append(StockPrice[idx: idx + max_len])
    this_day.append(StockPrice[idx + max_len])

x = np.array(historical_days)
y = np.array(this_day)


##Tran/test split
split = int(len(dataset) * 0.8)  
X_train = x[:split]
X_test = x[split:]
y_train = y[:split]
y_test = y[split:]


#splitting data into training &testing

data_training =pd.DataFrame(dataset['Close'][0:int(len(dataset)*0.70)])
data_testing =pd.DataFrame(dataset['Close'][int(len(dataset)*0.70):int(len(dataset))])


#performance 

def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(sum((y_pred-y_true)**2)/len(y_true)) 

def mean_absoulate_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_pred-y_true) / y_true))



        from sklearn.metrics import make_scorer
def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score
rmse_score = make_scorer(rmse, greater_is_better=False)

from sklearn.metrics import make_scorer
def mape(actual, predict):
    score = mean_absoulate_percentage_error(actual, predict)
    return score
mape_score = make_scorer(mape, greater_is_better=False)


#load Model

filename = 'Adaboost'
loaded_model = pickle.load(open(filename,'rb'))

loaded_model.fit(X_train, y_train)
y_pred_Ada = loaded_model.predict(X_test)

st.subheader('Graph of using ADABOOST Model')

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
fig = plt.figure(figsize=(12,6))

plt.plot(y_test, marker='.', label="Actual Price")
plt.plot(y_pred_Ada, 'r', label="Ada Boost prediction")
plt.ylabel('Stock Price ')
plt.xlabel('Time Step')
plt.title('Ada Boost Forecasting for  Price ')
plt.legend()
st.pyplot(fig)


future_next_value =loaded_model.predict(X_test[-1:])

st.text_area('Next Days Closing Predicted Value :',future_next_value)

# Make a prediction for the next value
prediction =loaded_model.predict(X_test)
predictions = []
 # Append the prediction to the list
predictions.append(prediction[0])

# Extract the predicted value for the 10th day
predicted_value_10th_day = predictions[-1]
st.text_area('Future  10th Day Closing Predicted Value :',predicted_value_10th_day);