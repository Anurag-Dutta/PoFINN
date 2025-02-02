import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from hampel import hampel  # Import for outlier detection (if needed)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot


# Function for shifting time series
def Supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # Prepare lag features for input
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # Combine all features into a dataframe
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # Drop NaN values if required
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def loss_fn(y_true, y_pred):
    epsilon = 1e-10  # Small value to avoid division by zero
    squared_difference = tf.square(y_true[:, 0] - y_pred[:, 0])

    squared_difference3 = tf.square(
        y_pred[:, 1] - (
            y_pred[:, 0] * (
                K0 - K1 * (
                    9 * a * tf.math.log((y_pred[:, 0] + epsilon) / C0) / (K0 - K1 * c)**2 +
                    4 * b * tf.math.log((y_pred[:, 0] + epsilon) / C0) / (K0 - K1 * c) + c
                )
            )
        )
    )
    return 0.8 * tf.reduce_mean(squared_difference, axis=-1) + 0.2 * tf.reduce_mean(squared_difference3, axis=-1)


if __name__ == '__main__':
    data = np.genfromtxt('capa_intermittency.dat')
    training_set = pd.DataFrame(data).reset_index(drop=True).iloc[:, 0]  # Only the first column
    t_diff = 1  # Time difference for gradient calculation
    gradient_t = (training_set.diff() / t_diff).iloc[1:]
    data = pd.read_csv("c1_interpolated_1250_100.csv")
    training_set = data.iloc[:, 1]
    test = training_set.tail(100)
    training_set = training_set.head(1250)
    training_set = training_set.reset_index(drop=True)
    gradient_t = gradient_t.reset_index(drop=True)

    # Combine training data and gradients into one dataframe
    df = pd.concat((training_set, gradient_t), axis=1)
    df.columns = ['y_t', 'grad_t']

    # Trainable parameters
    C0 = tf.Variable(86.6465, name="C0", trainable=True, dtype=tf.float32)
    K0 = tf.Variable(-0.0029, name="K0", trainable=True, dtype=tf.float32)
    K1 = tf.Variable(-0.0003, name="K1", trainable=True, dtype=tf.float32)
    a = tf.Variable(0.0000, name="a", trainable=True, dtype=tf.float32)
    b = tf.Variable(0.0168, name="b", trainable=True, dtype=tf.float32)
    c = tf.Variable(2.3581, name="c", trainable=True, dtype=tf.float32)
    L = np.minimum(C0, (df.iloc[:, 1] - (df.iloc[:, 0] * (K0 - K1 * (9 * a * np.log(df.iloc[:, 0] / C0) / (K0 - K1 * c)**2 + 4 * b * np.log(df.iloc[:, 0] / C0) / (K0 - K1 * c) + c))))

    # Generate supervised data for training
    data = Supervised(df.values, n_in=350, n_out=100)

    # Drop unnecessary columns based on lag values
    cols_to_drop = []
    for i in range(2, 351):
        cols_to_drop.extend([f'var2(t-{i})'])
    data.drop(cols_to_drop, axis=1, inplace=True)

    train = np.array(data[0:len(data)-1])
    forecast = np.array(data.tail(1))
    trainy = train[:, -300:]
    trainX = train[:, :-300]
    forecasty = forecast[:, -300:]
    forecastX = forecast[:, :-300]
    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    forecastX = forecastX.reshape((forecastX.shape[0], 1, forecastX.shape[1]))
    
    splitr = 0.8
    model = Sequential()
    model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))  # LSTM layer
    model.add(Dense(60))
    model.compile(loss=loss_fn, optimizer='adam')  # Compile model with custom loss function (phy + data)
    history = model.fit(trainX[:int(splitr*trainX.shape[0])], trainy[:int(splitr*trainX.shape[0])],
                        epochs=500, batch_size=64,
                        validation_data=(trainX[int(splitr*trainX.shape[0]):trainX.shape[0]],
                                         trainy[int(splitr*trainX.shape[0]):trainX.shape[0]]), shuffle=False)

    forecast_without_mc = forecastX
    yhat_without_mc = model.predict(forecast_without_mc)
    forecast_without_mc = forecast_without_mc.reshape((forecast_without_mc.shape[0], forecast_without_mc.shape[2]))
    inv_yhat_without_mc = np.concatenate((forecast_without_mc, yhat_without_mc), axis=1)
    fforecast = inv_yhat_without_mc[:, -300:]
    final_forecast = fforecast[:, 0:300:3]
    final_forecast[final_forecast < 0] = 0  # Set negative values to zero

    # Prints the final forecast
    final_forecast

    training_set = np.array(training_set)
    test = np.array(test)
    final_forecast = np.array(final_forecast.squeeze(0))
    MSE = np.square(np.subtract(np.array(test), np.array(final_forecast))).mean()
    rsme = math.sqrt(MSE)  # Root Mean Squared Error (RSME)
    print(rsme)
    MAE = np.abs(np.subtract(np.array(test), np.array(final_forecast))).mean()
    mae = MAE
    print(mae)
