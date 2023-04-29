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
# from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
from scipy import stats
from scipy.stats import mannwhitneyu 


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import pickle
import streamlit as st
import yfinance as yf
import datetime
import pickle
import streamlit as st
import yfinance as yf
import datetime as dt
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
import math # Mathematical functions 
import numpy as np # Fundamental package for scientific computing with Python
import pandas as pd # Additional functions for analysing and manipulating data
from datetime import date, timedelta, datetime # Date Functions
from pandas.plotting import register_matplotlib_converters # This function adds plotting functions for calender dates
import matplotlib.pyplot as plt # Important package for visualization - we use this to plot the market data
import matplotlib.dates as mdates # Formatting dates
from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance / errors
from keras.models import Sequential # Deep learning library, used for neural networks
from keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers
from keras.callbacks import EarlyStopping # EarlyStopping during model training
from sklearn.preprocessing import RobustScaler, MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data 
import seaborn as sns

import datetime
st.subheader('Query parameters')


start_date = st.date_input("Start date", datetime.date(2021, 1, 1))
end_date = st.date_input("End date", datetime.date(2023, 12, 31))

# Retrieving tickers data
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
tickerSymbol = st.selectbox('Stock ticker', ticker_list) # Select ticker symbol
tickerData = yf.Ticker(tickerSymbol) # Get ticker data
data = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker






st.subheader('STOCK PEICE LIST')
# if your dataset contains missing value, check which column has missing values
data.isnull().sum()





StockPrice = data['Close']











st.write(data.tail(10))
df=data
train_df = df.copy()
# We safe a copy of the dates index, before we need to reset it to numbers
date_index = train_df.index

# Adding Month and Year in separate columns

d = pd.to_datetime(train_df.index)
train_df['Month'] = d.strftime("%m").astype(int)
train_df['Year'] = d.strftime("%Y").astype(int)
train_df['Day'] = d.strftime("%d").astype(int)
# We reset the index, so we can convert the date-index to a number-index
#train_df = train_df.reset_index(drop=True).copy()
FEATURES = ['Open', 'High', 'Low', 'Close', 
                'Volume','Dividends']#, 'USAMoneySupplyM'
# Create the dataset with features and filter the data to the list of FEATURES
data = pd.DataFrame(train_df)
data_filtered = data[FEATURES]
data_filtered.head(10)
# We add a prediction column and set dummy values to prepare the data for scaling
#data_filtered_ext = data_filtered.copy()
#data_filtered_ext['Prediction'] = data_filtered_ext['Price']

# Print the tail of the dataframe
#data_filtered_ext.tail()

nrows = data_filtered.shape[0]

# Convert the data to numpy values
np_data_unscaled = np.array(data_filtered)
np_data = np.reshape(np_data_unscaled, (nrows, -1))
print(np_data.shape)

# Transform the data by scaling each feature to a range between 0 and 1
scaler = MinMaxScaler()
np_data_scaled = scaler.fit_transform(data_filtered)
df_scaled = pd.DataFrame(np_data_scaled)
# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = MinMaxScaler()
df_Close = pd.DataFrame(df['Close'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)

sequence_length =5 ###5

# Prediction Index
index_Close = data.columns.get_loc("Close")

# Split the training data into train and train data sets
# As a first step, we get the number of rows to train the model on 80% of the data 
train_data_len = math.ceil(np_data_unscaled.shape[0] * 0.8)

# Create the training and test data
train_data = np_data_scaled[0:train_data_len, :]
test_data = np_data_scaled[train_data_len - sequence_length:, :]

# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, sequence_length time steps per sample, and 6 features
def partition_dataset(sequence_length, data):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(sequence_length, data_len):
        x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction
    
    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

# Generate training data and test data
x_train, y_train = partition_dataset(sequence_length, train_data)
x_test, y_test = partition_dataset(sequence_length, test_data)

import tensorflow 
import keras

from tensorflow.keras.models import load_model
model = load_model('lstm_model.h5')


# Compile the model
model.compile(optimizer='adam', loss='mse')
epochs =  100#10
batch_size = 16
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history = model.fit(x_train, y_train, 
                    batch_size=batch_size, 
                    epochs=epochs,
                    validation_data=(x_test, y_test)
                   )

# Get the predicted values
y_pred_scaled = model.predict(x_test)

# Unscale the predicted values
y_pred = scaler_pred.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

# # Root Mean Squared Error (RMSE)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f'Root Mean Squared Error (RMSE): {rmse}')#np.round(rmse, 2)

# #RMSE
# print('Root Mean Squared Error (RMSE): %.4f'% np.sqrt(sum((y_pred-y_test_unscaled)**2)/len(y_test)))

# # Mean Absolute Error (MAE)
# MAE = mean_absolute_error(y_test_unscaled, y_pred)
# print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

# # Mean Absolute Percentage Error (MAPE)
# MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
# print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

# # Median Absolute Percentage Error (MDAPE)
# MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
# print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

#RMSE
rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred))
# rmse = np.sqrt(mse)
#print("Root Mean Squared Error (RMSE): {:.4f}".format(rmse))

# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
#print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

#R-squared
r2 = r2_score(y_test_unscaled, y_pred)
#print("R-squared score (R^2): {:.4f}".format(r2))


from datetime import datetime as dt
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
fig = plt.figure(figsize=(12,6))

plt.plot(y_test_unscaled, marker='.', label="Actual Price")
plt.plot(y_pred, 'r', label="LSTM prediction")
plt.ylabel(' Price  ')
plt.xlabel('Time Step')
plt.title('LSTM Forecasting for  Stock Price ')
plt.legend()
st.pyplot(fig)
st.subheader('Next 10 days Predicted Closing Price')
# Get the last date in the dataset
last_date = df.index[-1]

# Create a list of dates for the next 10 days
prediction_dates = pd.date_range(last_date, periods=10, freq='D')[1:]

# Generate predictions for the next 10 days
prediction_list = y_pred[-9:]

# Create a DataFrame to store the predictions
predictions_df = pd.DataFrame({'Date': prediction_dates, 'Predicted Price': np.ravel(prediction_list)})

# Set the date column as the index
predictions_df.set_index('Date', inplace=True)
# Reset the index before displaying the DataFrame in the table
predictions_df.reset_index(inplace=True)

# Display the predictions in a table
st.table(predictions_df)
