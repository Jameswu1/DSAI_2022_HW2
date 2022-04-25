import pandas as pd
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
#from pmdarima import auto_arima
import warnings
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#from nowcast_lstm.LSTM import LSTM
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from fbprophet import Prophet
from matplotlib import pyplot as plt
import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING) 
import argparse

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    #讀資料 並整理成有時間的序列
    colnames=['open','high','low','close']
    df_temp = pd.read_csv( args.training, names=colnames)
    df_temp_1 = pd.read_csv( args.testing, names=colnames)
    df = pd.concat([df_temp,df_temp_1])
    df = df.reset_index()
    df = df.loc[:,["open","high","low","close"]]
    date = pd.date_range(start='2015/01/01',periods=len(df))
    date = pd.DataFrame(date)
    df[['date']] = date

    #透過增加特徵的方式使模型訓練能得到更多資訊
    def relative_strength_idx(df, n=14):
        close = df['close']
        delta = close.diff()
        delta = delta[1:]
        pricesUp = delta.copy()
        pricesDown = delta.copy()
        pricesUp[pricesUp < 0] = 0
        pricesDown[pricesDown > 0] = 0
        rollUp = pricesUp.rolling(n).mean()
        rollDown = pricesDown.abs().rolling(n).mean()
        rs = rollUp / rollDown
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    # SMA
    df['EMA_3'] = df['close'].ewm(3).mean().shift()
    df['EMA_7'] = df['close'].ewm(7).mean().shift()
    df['EMA_30'] = df['close'].ewm(30).mean().shift()

    # EMA
    df['SMA_3'] = df['close'].rolling(3).mean().shift()
    df['SMA_7'] = df['close'].rolling(7).mean().shift()
    df['SMA_30'] = df['close'].rolling(30).mean().shift()

    # RSI
    df['RSI'] = relative_strength_idx(df).fillna(0)

    # MACD
    EMA_12 = pd.Series(df['close'].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(df['close'].ewm(span=26, min_periods=26).mean())
    df['MACD'] = pd.Series(EMA_12 - EMA_26)
    df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())


    df['y'] = df['open'].shift(-1)
    df = df.dropna(axis=0).reset_index(drop=True)

    #模型訓練
    features = ['SMA_3','SMA_7','SMA_30','EMA_3','EMA_7','EMA_30','RSI','MACD','MACD_signal']
    df_train = df[:-len(df_temp_1)]
    df_valid = df[-len(df_temp_1):]
    model_fbp = Prophet()
    for feature in features:
        model_fbp.add_regressor(feature)

    model_fbp.fit(df_train[["date", "y"] + features].rename(columns={"date": "ds", "y": "y"}))
    forecast = model_fbp.predict(df_valid[["date", "y"] + features].rename(columns={"date": "ds"}))
    df_valid["Forecast_Prophet"] = forecast.yhat.values

    #轉成list進行投資策略的計算
    ans = df_valid["Forecast_Prophet"].values.tolist()
    ans_l = [] 
    tmp = 0
    for i in range(len(ans)-2):
        if ans[i+2]>ans[i+1]:
            if tmp != 1:
                ans_l.append(1)
                tmp += 1 
            else:
                ans_l.append(0)
        else:
            if tmp != -1:
                ans_l.append(-1)
                tmp += -1 
            else:
                ans_l.append(0)
    if tmp == 0:
        ans_l.append(0)
    elif tmp == 1:
        ans_l.append(-1)
    elif tmp == -1:
        ans_l.append(1)

    #存檔
    test = pd.DataFrame(data=ans_l)
    test.to_csv("output.csv")


