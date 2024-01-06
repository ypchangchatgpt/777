import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import mpltw

# Streamlit 應用標題
st.title('股票模型分析及預測分析')

# 讓用戶選擇股票代碼和日期範圍
stock_symbol = st.text_input('請輸入股票代碼，例如 "0050.TW"', '0050.TW')
start_date = st.text_input('模型開始日期，格式：YYYY-MM-DD', '2019-01-01')
end_date = st.text_input('模型結束日期，格式：YYYY-MM-DD', '2021-12-31')
start_date1 = st.text_input('預測開始日期，格式：YYYY-MM-DD', '2022-01-01')
end_date1 = st.text_input('預測結束日期，格式：YYYY-MM-DD', '2023-12-31')

# 讓用戶自定義參數
mean_options = st.multiselect('選擇均值模型', ['AR', 'Constant', 'Zero','LS','ARX','HAR','HARX'], default=['AR'])
dist_options = st.multiselect('選擇分佈', ['normal', 't', 'skewt','gaussian','studentst','skewstudent','ged','generalized error'], default=['normal'])
max_lags = st.number_input('設定最大滯後項', min_value=0, max_value=10, value=3)
max_p = st.number_input('設定最大自回歸項', min_value=0, max_value=10, value=3)
max_q = st.number_input('設定最大移動平均項', min_value=0, max_value=10, value=3)
max_o = st.number_input('設定最大不對稱項', min_value=0, max_value=10, value=3)

# 當用戶輸入所有信息後運行分析
if st.button('開始分析'):
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
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lb_test.index, lb_test['lb_pvalue'], marker='o', linestyle='None')
    ax.axhline(y=0.05, color='r', linestyle='--', label='顯著性水平：0.05')
    ax.set_title('Ljung-Box 檢定')
    ax.set_xlabel('滯後期數量')
    ax.set_ylabel('p-value')
    ax.legend()
    st.pyplot(fig)
    if (lb_test['lb_pvalue'].dropna() > 0.05).all():
        st.write('殘差符合AR模型假設。')
    else:
        st.write('殘差不符合AR模型假設。')
    #Ljung-Box 檢定（殘差平方）
    lb_test_squared = sm.stats.acorr_ljungbox(residuals_std**2, lags=lags, model_df=best_lags)
    st.subheader('Ljung-Box 檢定（殘差平方）')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lb_test_squared.index, lb_test_squared['lb_pvalue'], marker='o', linestyle='None')
    ax.axhline(y=0.05, color='r', linestyle='--', label='顯著性水平：0.05')
    ax.set_title('Ljung-Box 檢定（殘差平方）')
    ax.set_xlabel('滯後期數量')
    ax.set_ylabel('p-value')
    ax.legend()
    st.pyplot(fig)
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
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 畫第一個數據集和設置 Y 軸
    ax1.set_xlabel('Time')
    ax1.set_ylabel('均值預測', color='black')
    ax1.plot(forecast['Mean Forecast'], color='green', label='均值預測')
    ax1.tick_params(axis='y', labelcolor='black')

    # 創建一個共享 X 軸但不同 Y 軸的第二個軸
    ax2 = ax1.twinx()  
    ax2.set_ylabel('波動率預測', color='black')
    ax2.plot(forecast['Vol_Forecast'], color='blue', label='波動率預測')
    ax2.tick_params(axis='y', labelcolor='black')

    # 添加標題和顯示圖例
    fig.tight_layout()  
    plt.title('均值預測 vs 波動率預測')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    st.pyplot(fig)



    
    data = pd.concat([data2, forecast], axis=1) 
    data.columns = ['Log_Return','Mean Forecast', 'Vol_Forecast']
    # 繪製預測結果
    st.subheader('均值預測 vs 波動率預測 VS 實際報酬')
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 畫第一個數據集和設置 Y 軸
    ax1.set_xlabel('Time')
    ax1.set_ylabel('報酬率％', color='black')
    ax1.plot(data['Mean Forecast'], color='blue', label='均值預測')
    ax1.plot(data['Log_Return'], color='green', label='實際報酬')
    ax1.tick_params(axis='y', labelcolor='black')

    # 創建一個共享 X 軸但不同 Y 軸的第二個軸
    ax2 = ax1.twinx()  
    ax2.set_ylabel('波動率預測', color='black')
    ax2.plot(data['Vol_Forecast'], color='red', label='波動率預測')
    ax2.tick_params(axis='y', labelcolor='black')

    # 添加標題和顯示圖例
    fig.tight_layout()  
    plt.title('均值預測 vs 波動率預測 VS 實際報酬')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    st.pyplot(fig)


    # 計算波動性指標
    ln_hl_squared = ((np.log(stock_data['High'] / stock_data['Low']))**2)
    parkinson_volatility = np.sqrt(ln_hl_squared / (4 * np.log(2)))
    parkinson_volatility = parkinson_volatility[-data2_n:] * 100
    stock_data['Parkinson_Volatility'] = parkinson_volatility
    data_volatility = pd.concat([parkinson_volatility, vol_forecast], axis=1) 
    data_volatility.columns = ['parkinson_volatility', 'vol_forecast']
    data_volatility['vol_forecast'] = data_volatility['vol_forecast'].shift(1)

    # 繪製波動性指標圖表
    st.subheader('波動率預測 vs 帕金森波動率')
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(data_volatility['vol_forecast'], color='blue', label='波動率預測')
    ax1.plot(data_volatility['parkinson_volatility'], color='green', label='帕金森波動率')
    ax1.set_title('波動率預測 vs 帕金森波動率')
    ax1.set_ylabel('%')
    ax1.legend()
    st.pyplot(fig1)

    # 實際值和預測值之走勢圖
    price = stock_data['Close']
    price = price[start_date1:]
    data_price = pd.concat([price, mean_forecast], axis=1) 
    data_price.columns = ['close', 'forecast']
    data_price['close_forecast'] = data_price['close'] / np.exp(data_price['forecast'] / 100)
    data_price['close_forecast'] = data_price['close_forecast'].shift(1)

    # 繪製實際價格與預測價格對比圖
    st.subheader('實際價格對比預測價格')
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(data_price['close_forecast'], color='blue', label='預測價格')
    ax2.plot(data_price['close'], color='green', label='實際價格')
    ax2.set_title('實際價格對比預測價格')
    ax2.set_ylabel('＄')
    ax2.legend()
    st.pyplot(fig2)
