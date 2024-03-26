#streamlit run /Users/lizongsiou/Desktop/數值分析/布朗運動/布朗運動.py

import streamlit as st
import numpy as np
from scipy.stats import qmc, norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# 設置 Streamlit 頁面的基本配置
st.set_page_config(
   page_title="查理布朗",  # 設定網頁標題
   page_icon='布朗的頭.png',  # 設定網頁圖標，路徑指向本地圖標文件
   layout="wide",  # 設定頁面布局為寬屏模式
   initial_sidebar_state="expanded" ) # 初始時側邊欄狀態為展開

# 在頁面頂部顯示一幅圖片
st.image('布朗運動.png', width=800)  # 圖片路徑及其寬度

# 顯示應用的標題
st.title("Geometric Brownian motion  模型下，請利用模擬方法得到歐式選擇權 (European option) 評價")
st.sidebar.image('布朗進化.png', width=300)
with st.sidebar.form(key='my_form'):
    option_type = st.radio("選擇權型態",("Call", "Put"),horizontal=True)
    # 初始股價的滑動選擇
    S0 = st.slider('選擇初始股價 S0', min_value=50, max_value=300, value=100)

    # 行使價格的滑動選擇
    K = st.slider('選擇行使價格 K', min_value=50, max_value=300, value=100)

    # 到期時間的滑動選擇（以年為單位）
    T = st.slider('選擇到期時間 T (年)', min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    # 無風險利率的滑動選擇
    r = st.slider('選擇無風險利率 r', min_value=0.01, max_value=0.10, value=0.03, step=0.01)

    # 波動率的滑動選擇
    sigma = st.slider('選擇波動率 σ', min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    # 模擬次數的滑動選擇
    n_simulations = st.slider('選擇模擬次數', min_value=1000, max_value=20000, value=10000, step=1000)

    # 實驗重複次數的滑動選擇
    n_experiments = st.slider('選擇實驗重複次數', min_value=100, max_value=5000, value=1000, step=100)
    seed = st.text_input(label="亂數種子", value="123457")
    submit_button = st.form_submit_button(label='查理布朗叫史努比去分析')

# 函數：計算歐式選擇權價格、到期股票價格並進行折現
def european_option_price(S0, K, T, r, sigma, Z):
    # 注意這裡使用 r 作為漂移率
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    call_payoff = np.maximum(ST - K, 0)
    put_payoff = np.maximum(K - ST, 0)
    call_price = np.exp(-r * T) * np.mean(call_payoff)
    put_price = np.exp(-r * T) * np.mean(put_payoff)
    return call_price, put_price, ST

# 初始化列表來收集數據
call_prices_pseudo, put_prices_pseudo, STs_pseudo = [], [], []
call_prices_quasi, put_prices_quasi, STs_quasi = [], [], []
call_prices_numpy, put_prices_numpy, STs_numpy = [], [], []    

if submit_button:
    bar = st.progress(20,'史努比：哭啊遇到一個爛主人')#顯示進度條
    
    np.random.seed(int(seed))
    sobol_engine = qmc.Sobol(d=1, scramble=True, seed=int(seed))
    for _ in range(n_experiments):
        # 使用偽隨機數
        pseudo_uniform = np.random.uniform(0, 1, n_simulations)
        pseudo_random_Z = norm.ppf(pseudo_uniform)
        call_price_pseudo, put_price_pseudo, ST_pseudo = european_option_price(S0, K, T, r, sigma, pseudo_random_Z)
        call_prices_pseudo.append(call_price_pseudo)
        put_prices_pseudo.append(put_price_pseudo)
        STs_pseudo.append(np.mean(ST_pseudo))
        
        # 使用拟隨機數
        quasi_random = sobol_engine.random(n=n_simulations).flatten()
        quasi_random_Z = norm.ppf(quasi_random)
        call_price_quasi, put_price_quasi, ST_quasi = european_option_price(S0, K, T, r, sigma, quasi_random_Z)
        call_prices_quasi.append(call_price_quasi)
        put_prices_quasi.append(put_price_quasi)
        STs_quasi.append(np.mean(ST_quasi))
        
        # 使用Numpy隨機數
        numpy_random_Z = np.random.normal(0, 1, n_simulations)
        call_price_numpy, put_price_numpy, ST_numpy = european_option_price(S0, K, T, r, sigma, numpy_random_Z)
        call_prices_numpy.append(call_price_numpy)
        put_prices_numpy.append(put_price_numpy)
        STs_numpy.append(np.mean(ST_numpy))
    bar = bar.progress(80,'史努比：計算中別催')   
    # Black-Scholes 公式的中間變量
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # 真實的看漲選擇權（Call）價格
    true_call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # 真實的看跌選擇權（Put）價格
    true_put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    



    
    
    if option_type == "Call":
        # 創建包含三個子圖的圖表
        fig = make_subplots(rows=3, cols=1, subplot_titles=('pseudo', 'quasi', 'Numpy'))
        st.subheader('看漲選擇權價格分布比較')
        # 添加偽隨機數據的直方圖到第一個子圖，並設置透明度
        fig.add_trace(go.Histogram(
            x=call_prices_pseudo, 
            name='pseudo', 
            marker_color='blue', 
            opacity=0.8  # 設置透明度
            ), row=1, col=1)
    
    # 添加 Black-Scholes 真值的垂直線
        fig.add_trace(go.Scatter(
            x=[true_call_price, true_call_price], 
            y=[0,100], 
            mode='lines', 
            name='Black-Scholes 真值', 
            line=dict(color='black')
            ,showlegend=False), row=1, col=1)
    
        # 重複上述步驟，對第二和第三個子圖進行相同的操作，並設置透明度
        fig.add_trace(go.Histogram(
            x=call_prices_quasi, 
            name='quasi', 
            marker_color='red', 
            opacity=0.8
            ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=[true_call_price, true_call_price], 
            y=[0,100], 
            mode='lines', 
            name='Black-Scholes 真值', 
            line=dict(color='black')
            ,showlegend=False), row=2, col=1)
        
        fig.add_trace(go.Histogram(
            x=call_prices_numpy, 
            name='Numpy', 
            marker_color='green', 
            opacity=0.8
            ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=[true_call_price, true_call_price], 
            y=[0,100], 
            mode='lines', 
            name='Black-Scholes 真值', 
            line=dict(color='black')
            ,showlegend=False), row=3, col=1)
        
            # 更新佈局設定
        fig.update_layout(
            title='不同方法計算的看漲選擇權價格分布',
            xaxis_title='看漲選擇權價格',
            yaxis_title='',
            height=900  # 設定圖表的高度，以便於展示所有子圖
            )
        # 顯示圖表
        st.plotly_chart(fig)
            
    if option_type == "Put":
        st.subheader('看跌選擇權價格分布比較')

        # 創建包含三個子圖的圖表
        fig = make_subplots(rows=3, cols=1, subplot_titles=('pseudo', 'quasi', 'Numpy'))
        # 添加偽隨機數據的直方圖到第一個子圖，並設置透明度
        fig.add_trace(go.Histogram(
            x=put_prices_pseudo, 
            name='pseudo', 
            marker_color='blue', 
            opacity=0.8  # 設置透明度
            ), row=1, col=1)
    
        # 添加 Black-Scholes 真值的垂直線
        fig.add_trace(go.Scatter(
            x=[true_put_price, true_put_price], 
            y=[0,100], 
            mode='lines', 
            name='Black-Scholes 真值', 
            line=dict(color='black')
            ,showlegend=False), row=1, col=1)
    
        # 重複上述步驟，對第二和第三個子圖進行相同的操作，並設置透明度
        fig.add_trace(go.Histogram(
            x=put_prices_quasi, 
            name='quasi', 
            marker_color='red', 
            opacity=0.8
            ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=[true_put_price, true_put_price], 
            y=[0,100], 
            mode='lines', 
            name='Black-Scholes 真值', 
        line=dict(color='black')
            ,showlegend=False), row=2, col=1)
        
        fig.add_trace(go.Histogram(
            x=put_prices_numpy, 
            name='Numpy', 
        marker_color='green', 
            opacity=0.8
            ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=[true_put_price, true_put_price], y=[0,100], 
            mode='lines', 
            name='Black-Scholes 真值', 
            line=dict(color='black')
            ,showlegend=False), row=3, col=1)
        
        # 更新佈局設定
        fig.update_layout(
            title='不同方法計算的看跌選擇權價格分布',
            xaxis_title='看跌選擇權價格',
            yaxis_title='',
            height=900  # 設定圖表的高度，以便於展示所有子圖
            )
        # 顯示圖表
        st.plotly_chart(fig)
    
    
    bar = bar.progress(100,'史努比：好了拉拿去')
                