#streamlit run /Users/lizongsiou/Desktop/untitled3.py

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import mpltw
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
# Streamlit 應用標題
st.title('股票模型分析及預測分析')


url = "https://www.taifex.com.tw/cht/9/futuresQADetail"
table = pd.read_html(url)
stocks1 = table[0].iloc[:,1:4]
stocks2 = table[0].iloc[:,5:8]
stocks1.columns = ["代號", "證券名稱", "市值佔大盤比重"]
stocks2.columns = ["代號", "證券名稱", "市值佔大盤比重"]
stocks1 = stocks1.dropna()
stocks2 = stocks2.dropna()
stocks1["代號"] = stocks1["代號"].astype(str)
stocks2["代號"] = [str(int(stocks_代號)) for stocks_代號 in stocks2["代號"]]
# stocks2["代號"] 有 .0。
stocks = pd.concat([stocks1, stocks2], axis=0)
stocks = stocks.reset_index(drop=True)
# stocks["市值佔大盤比重"] 最後為 % 符號。
stocks["市值佔大盤比重"] = stocks["市值佔大盤比重"].str[:-1].astype(float)/100
stocks["代號"] = [stocks["代號"][i]+".TW" for i in range(len(stocks))]
stocks["代號_證券名稱_市值佔大盤比重"] = [stocks["代號"][i]+" "+stocks["證券名稱"][i]+" "+str(round(stocks["市值佔大盤比重"][i], 6)) for i in range(len(stocks))]


stock_ticker_name = st.sidebar.selectbox(label="請輸入股票代號_證券名稱_市值佔大盤比重",
                                         options=stocks["代號_證券名稱_市值佔大盤比重"])
stock_symbol  = stock_ticker_name.split(" ")[0]

start_date = st.sidebar.date_input(label="模型開始日期",
                              value=dt.date(2020, 1, 1),
                              format="YYYY-MM-DD")
end_date = st.sidebar.date_input(label="模型結束日期",
                            value=dt.date(2021,12,31),
                            # value="today",
                            format="YYYY-MM-DD")
start_date1 = st.sidebar.date_input(label="預測開始日期",
                              value=dt.date(2022, 1, 1),
                              format="YYYY-MM-DD")
end_date1 = st.sidebar.date_input(label="預測結束日期",
                            value=dt.date(2023,12,31),
                            # value="today",
                            format="YYYY-MM-DD")

# 讓用戶自定義參數
mean_options = st.sidebar.multiselect('選擇均值模型', ['AR', 'Constant', 'Zero','LS','ARX','HAR','HARX'], default=['AR'])
dist_options = st.sidebar.multiselect('選擇分佈', ['normal', 't', 'skewt','gaussian','studentst','skewstudent','ged','generalized error'], default=['normal'])
max_lags = st.sidebar.number_input('設定最大滯後項(Lags)', min_value=0, max_value=10, value=3)
max_p = st.sidebar.number_input('設定最大自回歸項(P)', min_value=0, max_value=10, value=3)
max_q = st.sidebar.number_input('設定最大移動平均項(Q)', min_value=0, max_value=10, value=3)
max_o = st.sidebar.number_input('設定最大不對稱項(O)', min_value=0, max_value=10, value=3)

# 當用戶輸入所有信息後運行分析
if st.sidebar.button('開始分析'):
    # 加載數據
    stock_data = yf.download(stock_symbol, start='2000-01-01', end=datetime.now(), auto_adjust=True)
    data_all = np.log(stock_data['Close']/stock_data['Close'].shift(1)).dropna()*100
    data1 = data_all[start_date:end_date]

    # 初始化模型選擇參數
    model_bic = []
    best_lags, best_dist, best_p, best_q, best_o, best_mean = None, None, None, None, None, None
    residuals, residuals_std = [], []

    # 遍歷不同的模型參數
    for mean in mean_options:
        for dist in dist_options:
            for lags in range(max_lags + 1):
                for p in range(max_p + 1):
                    for q in range(max_q + 1):
                        for o in [0, min(p, max_o)]: 
                            if p == 0 and q == 0:
                                model = arch_model(data1, mean=mean, lags=lags, vol="Constant", p=p, q=q, o=o, dist=dist)
                            elif p > 0 or o > 0:
                                model = arch_model(data1, mean=mean, lags=lags, vol='Garch', p=p, q=q, o=o, dist=dist)
                            else:
                                continue

                            model_fit = model.fit(disp="off")
                            model_bic.append(model_fit.bic)
                            residualss = model_fit.resid
                            residuals2 = residualss / model_fit.conditional_volatility
                            lb_test = sm.stats.acorr_ljungbox(residuals2.dropna(), lags=20, model_df=lags+p+q+o, return_df=True)
                            lb_test_squared = sm.stats.acorr_ljungbox((residuals2.dropna())**2, lags=20, model_df=lags+p+q+o, return_df=True)

                            if (model_fit.bic == np.min(model_bic)) and (lb_test["lb_pvalue"].dropna() > 0.05).all() and (lb_test_squared["lb_pvalue"].dropna() > 0.05).all():
                                best_mean = mean
                                best_p = p
                                best_q = q
                                best_o = o
                                best_lags = lags
                                best_dist = dist
                                residuals = residualss
                                residuals_std = residuals2

    if best_mean is not None:
        st.subheader('最佳模型參數：')
        st.subheader(f'均值模型: {best_mean}')
        st.subheader(f'滯後項: {best_lags}')
        st.subheader(f'分佈: {best_dist}')
        st.subheader(f'P(自回歸項): {best_p}')
        st.subheader(f'Q(移動平均項) : {best_q}')
        st.subheader(f'O (不對稱項): {best_o}')
        st.subheader('模型摘要：')
        st.write(model_fit.summary())
    else:
        st.write('沒有找到符合條件的模型。')
        
    #Ljung-Box 檢定
    lags = 20  
    lb_test = sm.stats.acorr_ljungbox(residuals_std, lags=lags, model_df=best_lags)
    # Ljung-Box 檢定的視覺化
    st.subheader('Ljung-Box 檢定')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lb_test.index, y=lb_test['lb_pvalue'], mode='markers', name='p-value'))
    fig.add_hline(y=0.05, line=dict(color='red', dash='dash'), annotation_text='顯著性水平：0.05')
    fig.update_layout(title='Ljung-Box 檢定', xaxis_title='lags', yaxis_title='p-value')
    st.plotly_chart(fig)
    if (lb_test['lb_pvalue'].dropna() > 0.05).all():
        st.write('殘差符合AR模型假設。')
    else:
        st.write('殘差不符合AR模型假設。')
    #Ljung-Box 檢定（殘差平方）
    lb_test_squared = sm.stats.acorr_ljungbox(residuals_std**2, lags=lags, model_df=best_lags)
    st.subheader('Ljung-Box 檢定（殘差平方）')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lb_test_squared.index, y=lb_test_squared['lb_pvalue'], mode='markers', name='p-value'))
    fig.add_hline(y=0.05, line=dict(color='red', dash='dash'), annotation_text='顯著性水平：0.05')
    fig.update_layout(title='Ljung-Box 檢定（殘差平方）', xaxis_title='lags', yaxis_title='p-value')
    st.plotly_chart(fig)
    if (lb_test_squared['lb_pvalue'].dropna() > 0.05).all():
        st.write('殘差平方符合AR模型假設。')
    else:
        st.write('殘差平方不符合AR模型假設。')
        
    # mean_forecast vol_forecast
    data2 = data_all[start_date1:end_date1]
    data2_n = len(data2)
    test_n = data2_n - 250
    rolling_window_size = 250
    forecast_horizon = 1
    mean_forecast = pd.DataFrame()    
    vol_forecast = pd.DataFrame()

    for i in range(data2_n - rolling_window_size):
        model = arch_model(data2[i:i+rolling_window_size], mean=best_mean, lags=best_lags, vol='Garch', p=best_p, q=best_q, o=best_o, dist=best_dist)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=forecast_horizon)
        mean_forecast0 = pred.mean.iloc[[0]]
        vol_forecast0 = np.sqrt(pred.variance).iloc[[0]]
        mean_forecast = pd.concat([mean_forecast, mean_forecast0], axis=0)
        vol_forecast = pd.concat([vol_forecast, vol_forecast0], axis=0) 

    forecast = pd.concat([mean_forecast, vol_forecast], axis=1)
    forecast.columns = ['Mean Forecast', 'Vol_Forecast']
    forecast['Mean Forecast'] = forecast['Mean Forecast'].shift(1)
    forecast['Vol_Forecast'] = forecast['Vol_Forecast'].shift(1)

    st.subheader('均值預測 vs 波動率預測')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Mean Forecast'], mode='lines', name='均值預測', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Vol_Forecast'], mode='lines', name='波動率預測', line=dict(color='blue'), yaxis="y2"))
    fig.update_layout(title='均值預測 vs 波動率預測', xaxis_title='Time', yaxis_title='報酬率', yaxis2=dict(title='%', overlaying='y', side='right'))
    st.plotly_chart(fig)


    
    data = pd.concat([data2, forecast], axis=1) 
    data.columns = ['Log_Return','Mean Forecast', 'Vol_Forecast']
    # 繪製預測結果
    st.subheader('均值預測 vs 波動率預測 VS 實際報酬')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Mean Forecast'], mode='lines', name='均值預測', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Log_Return'], mode='lines', name='實際報酬率', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Vol_Forecast'], mode='lines', name='波動率預測', line=dict(color='red'), yaxis="y2"))
    fig.update_layout(title='均值預測 vs 波動率預測 VS 實際報酬', xaxis_title='Time', yaxis_title='%', yaxis2=dict(title='%', overlaying='y', side='right'))
    st.plotly_chart(fig)


    # 計算波動性指標
    real_volatility = np.sqrt(data2**2)
    data_volatility = pd.concat([real_volatility, vol_forecast], axis=1) 
    data_volatility.columns = ['real_volatility', 'vol_forecast']
    data_volatility = data_volatility.dropna()
    # 繪製波動性指標圖表
    st.subheader('波動率預測 vs 真實波動值')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_volatility.index, y=data_volatility['vol_forecast'], mode='lines', name='波動率預測', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data_volatility.index, y=data_volatility['real_volatility'], mode='lines', name='真實波動值', line=dict(color='green')))
    fig.update_layout(title='波動率預測 vs 真實波動值',xaxis_title='Time',yaxis_title='%')
    # 使用 Streamlit 顯示圖表
    st.plotly_chart(fig)

    # 計算平均絕對誤差 (MAE)
    mae = np.mean(np.abs(data_volatility['vol_forecast'] - data_volatility['real_volatility']))
    st.write('平均絕對誤差為：', mae)



    # 實際值和預測值之走勢圖
    price = stock_data['Close']
    price = price[start_date1:]
    data_price = pd.concat([price, mean_forecast], axis=1) 
    data_price.columns = ['close', 'forecast']
    data_price['close_forecast'] = data_price['close'] / np.exp(data_price['forecast'] / 100)
    data_price['close_forecast'] = data_price['close_forecast'].shift(1)

    # 繪製實際價格與預測價格對比圖
    st.subheader('實際價格對比預測價格')
    # 創建 Plotly 圖形對象
    fig2 = go.Figure()

    # 添加預測價格曲線
    # 使用 go.Scatter 創建曲線，mode='lines' 表示畫線圖
    fig2.add_trace(go.Scatter(x=data_price.index, y=data_price['close_forecast'], mode='lines', name='預測價格', line=dict(color='blue')))

    # 添加實際價格曲線
    fig2.add_trace(go.Scatter(x=data_price.index, y=data_price['close'], mode='lines', name='實際價格', line=dict(color='green')))

    # 更新圖表佈局
    # 設置圖表的標題、X 軸和 Y 軸的標籤
    fig2.update_layout(
    title='實際價格對比預測價格',
    xaxis_title='Time',
    yaxis_title='＄')

    # 使用 Streamlit 顯示圖表
    st.plotly_chart(fig2)