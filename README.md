# DSAI_2022_HW2
這次的作業內容主要是透過股市去預測未來20天的投資策略
針對這次的作業，我主要希望先透過各種指標的特徵去計算出隔天的預測開盤價格
再透過簡易的演算法針對漲跌去進行投資策略

## 特徵挑選
針對資料集的特徵主要包含了以下四種\
open - 開盤價格\
high - 最高點\
low - 最低點\
close - 關盤價格\
\
針對OHLC的指標，我們希望透過特徵的提取方式，得到這次想學習的特徵，也就是滯後指標（Lagging Indicator），包含以下特徵\
SMA - 一段期間的股價計算平均值\
EMA - 根據日期指數遞減加權，計算移動平均值，日期愈近的比重愈高\
RSI - 觀測買賣超指標\
MACD - 透過EMA觀測走勢量\
\
## 模型選擇
針對這次的任務，一樣透過兩種模型進行訓練\
都是經由時間序列的模型，第一個是ARIMA，第二個則是Prophet\
運行結果以Prophet以較高的分數取的優勢，因此本篇也單單介紹Prophet\
\
針對Prophet的模型，我們透過上述所選定的特徵值來進行運算\
以下則是針對運行所得到的趨勢\

![image](https://user-images.githubusercontent.com/41716487/164413720-5ce053c4-58fd-428e-b801-ae3d6696ac20.png)
![image](https://user-images.githubusercontent.com/41716487/164413778-1ebd6319-c2f1-4eac-888a-355eb0a693f9.png)
![image](https://user-images.githubusercontent.com/41716487/164413805-91fe0ab9-44df-45e0-a878-37a9fe94d819.png)
![image](https://user-images.githubusercontent.com/41716487/164413837-06db202a-c590-4460-bf97-58e6199755e1.png)

## 演算法
針對預測價格的數值之後，為了在最短的交易日中，以最大次數的交易來達到最佳值\
因此演算法選擇相對簡易的計算，分別去計算後兩天價格以及後一天價格\
假設day[i+1]>day[i]\
代表我能在預測時交易為正就買，賺到錢，因此就買進\
與之相反，如果不行，則進行賣出的動作。\
類似於naive的交易策略。\
