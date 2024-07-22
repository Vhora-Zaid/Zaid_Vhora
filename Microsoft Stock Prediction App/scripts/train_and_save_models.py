import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from statsmodels.tsa.arima.model import ARIMA

# Load your data
df = pd.read_csv('data/prices-split-adjusted.csv')

# Preprocess your data
X = df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']]
y = df['target']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data if necessary
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 1. Linear Regression
LR = LinearRegression()
LR.fit(x_train, y_train)
y_train_pred_LR = LR.predict(x_train)
y_pred_LR = LR.predict(x_test)
print("Linear Regression Test Prediction MSE:", mean_squared_error(y_test, y_pred_LR))
print("Linear Regression Train Prediction MSE:", mean_squared_error(y_train, y_train_pred_LR))
print("Linear Regression R2 Score:", r2_score(y_test, y_pred_LR))
print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_LR))
print("Linear Regression MAPE:", mean_absolute_percentage_error(y_test, y_pred_LR))

# Save Linear Regression model
joblib.dump(LR, 'models/linear_regression.pkl')

# 2. Random Forest
RF = RandomForestRegressor()
RF.fit(x_train, y_train)
y_train_pred_RF = RF.predict(x_train)
y_pred_RF = RF.predict(x_test)
print("Random Forest Test Prediction MSE:", mean_squared_error(y_test, y_pred_RF))
print("Random Forest Train Prediction MSE:", mean_squared_error(y_train, y_train_pred_RF))
print("Random Forest R2 Score:", r2_score(y_test, y_pred_RF))
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_RF))
print("Random Forest MAPE:", mean_absolute_percentage_error(y_test, y_pred_RF))

# Save Random Forest model
joblib.dump(RF, 'models/random_forest.pkl')

# 3. XGBoost
xg = XGBRegressor()
xg.fit(x_train, y_train)
y_train_pred_xg = xg.predict(x_train)
y_pred_xg = xg.predict(x_test)
print("XGBoost Test Prediction MSE:", mean_squared_error(y_test, y_pred_xg))
print("XGBoost Train Prediction MSE:", mean_squared_error(y_train, y_train_pred_xg))
print("XGBoost R2 Score:", r2_score(y_test, y_pred_xg))
print("XGBoost MAE:", mean_absolute_error(y_test, y_pred_xg))
print("XGBoost MAPE:", mean_absolute_percentage_error(y_test, y_pred_xg))

# Save XGBoost model
joblib.dump(xg, 'models/xgboost.pkl')

# 4. Simple ANN
model = Sequential()
model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save Simple ANN model
model.save('models/simple_ann.h5')

# 5. LSTM
model_LSTM = Sequential()
model_LSTM.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model_LSTM.add(LSTM(50))
model_LSTM.add(Dense(1))

model_LSTM.summary()
model_LSTM.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
history_LSTM = model_LSTM.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save LSTM model
model_LSTM.save('models/lstm.h5')

# 6. ARIMA
train = df.iloc[:500000, :]
test = df.iloc[500001:, :]

model_ARIMA = ARIMA(train['close'], order=(5, 1, 0))
model_ARIMA = model_ARIMA.fit()
start = len(train)
end = len(train) + len(test) - 1
predictions_ARIMA = model_ARIMA.predict(start=start, end=end, typ='levels')

# Save ARIMA model
model_ARIMA.save('models/arima.pkl')

# Visualize predictions
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['close'], label='Close Price')
plt.plot(y_pred_LR, label='Linear Regression Predictions')
plt.plot(y_pred_RF, label='Random Forest Predictions')
plt.plot(y_pred_xg, label='XGBoost Predictions')
plt.legend()
plt.show()
