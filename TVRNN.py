import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

filepath = ".csv"
df = pd.read_csv(filepath)

# 確認數值的大致樣貌，決定是否做預處理，大約從整體分佈來看僅需要做normalize來加速收斂即減少變數之間的趨勢

print(df['SPY'].describe())

# 設定目標變數
df['y'] = df['SPY']

# 資料處理的部份我先對整體數值做normalize


def normalize(df):
    nor_df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return nor_df


df.iloc[:, 1:] = normalize(df.iloc[:, 1:])

# 並拆出訓練集與測試集


def split_train_test(df, splitElement):

    i = df.Date.tolist().index(splitElement)
    train_data = df.iloc[:i, :]
    test_data = df.iloc[i:, :]
    return train_data, test_data


train_data, test_data = split_train_test(df, "2015-01-02")


train_data.drop(['Date'], axis=1, inplace=True)
test_data.drop(['Date'], axis=1, inplace=True)

# 根據題目所述，我將利用一週五天的股票收盤價來做預測下禮拜第一天的股票漲跌，故討論每五天預測第六天的模型。


def splitData(df):
    X_train = []
    Y_train = []
    days = 5
    pre_day = 1
    for i in range(df.shape[0] - days - pre_day):
        X_train.append(np.array(df.iloc[i:i + days, :-1]))
        Y_train.append(np.array(df.iloc[i + days:i + days + pre_day]['y']))

    return np.array(X_train), np.array(Y_train)


X_train, Y_train = splitData(train_data)
X_test, Y_test = splitData(test_data)

# 使用的模型即是RNN的LSTM，並利用dropout0.2來避免overfitting。


def lstm_model(train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(train.shape[1], train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model


# 使用mean squared error來判定預測股價與真實值的差距
model = lstm_model(X_train)

callback = EarlyStopping(monitor="mean_squared_error", patience=10, verbose=1, mode="auto")
history = model.fit(X_train, Y_train, epochs=100, batch_size=10, validation_split=0.1, callbacks=[callback], shuffle=True)

# 帶入測試集
model.evaluate(X_test, Y_test)

predict_y = model.predict(X_test)

# 實際來看每隔週一的漲跌，與預測結果的差距


def updown(array, days, pre_day):
    updown = []

    for i in range(array.shape[0] - days - pre_day):
        if array[i][0] <= array[i + days + pre_day][0]:
            updown.append("up")
        else:
            updown.append("down")
    return updown


updown_test = updown(Y_test, 5, 1)
updown_pred = updown(predict_y, 5, 1)

# 計算預測準確度


def accuracy(test, pred):
    right = 0
    for i in range(len(test)):
        if test[i] == pred[i]:
            right += 1
        else:
            right = right

    accuracy = right / len(test)

    return accuracy


print(accuracy(updown_test, updown_pred))

plt.subplot(211)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.subplot(212)

plt.plot(Y_test, color='red', label='Real Price')
plt.plot(predict_y, color='blue', label='Predicted Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
