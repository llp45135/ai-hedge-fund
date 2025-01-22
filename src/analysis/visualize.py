"""
技术分析可视化模块

该模块提供了一系列用于可视化技术分析指标的函数，
包括趋势分析、成交量分析、动量指标等的可视化。
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from agents.technicals import (
    calculate_ema,
    calculate_volume_ema,
    calculate_adx,
    calculate_volume_rsi,
    calculate_trend_signals
)

def visualize_trend_signals(prices_df: pd.DataFrame, window_size: int = 200) -> None:
    """
    可视化趋势信号分析，创建包含价格、成交量、ADX和RSI的多子图分析面板。
    
    Args:
        prices_df: 包含OHLCV数据的DataFrame，需要包含以下列：
                  - open: 开盘价
                  - high: 最高价
                  - low: 最低价
                  - close: 收盘价
                  - volume: 成交量
        window_size: 显示的数据窗口大小，默认200个交易日
        
    Returns:
        None: 直接显示交互式图表
    """
    # 计算技术指标
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)
    
    volume_ema_5 = calculate_volume_ema(prices_df, 5)
    volume_ema_20 = calculate_volume_ema(prices_df, 20)
    
    adx_df = calculate_adx(prices_df, 14)
    volume_rsi = calculate_volume_rsi(prices_df)
    
    # 创建子图布局
    fig = make_subplots(
        rows=4, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=(
            '价格趋势分析', 
            '成交量分析',
            'ADX趋势强度',
            '成交量RSI'
        )
    )

    # 1. 主图表：价格和EMAs
    fig.add_trace(
        go.Candlestick(
            x=prices_df.index,
            open=prices_df['open'],
            high=prices_df['high'],
            low=prices_df['low'],
            close=prices_df['close'],
            name='价格'
        ),
        row=1, col=1
    )
    
    # 添加EMAs
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=ema_8,
            name='EMA 8',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=ema_21,
            name='EMA 21',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=ema_55,
            name='EMA 55',
            line=dict(color='red', width=1)
        ),
        row=1, col=1
    )

    # 2. 成交量分析
    fig.add_trace(
        go.Bar(
            x=prices_df.index,
            y=prices_df['volume'],
            name='成交量',
            marker_color='rgba(128,128,128,0.5)'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=volume_ema_5,
            name='成交量EMA 5',
            line=dict(color='blue', width=1)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=volume_ema_20,
            name='成交量EMA 20',
            line=dict(color='orange', width=1)
        ),
        row=2, col=1
    )

    # 3. ADX趋势强度
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=adx_df['adx'],
            name='ADX',
            line=dict(color='purple', width=1)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=adx_df['+di'],
            name='+DI',
            line=dict(color='green', width=1)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=adx_df['-di'],
            name='-DI',
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )

    # 4. 成交量RSI
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=volume_rsi,
            name='成交量RSI',
            line=dict(color='blue', width=1)
        ),
        row=4, col=1
    )
    
    # 添加RSI参考线
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    
    # 更新布局
    fig.update_layout(
        title='趋势信号分析面板',
        xaxis_title='日期',
        height=1200,  # 增加图表高度
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # 更新Y轴标签
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="ADX/DI", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)
    
    # 显示最近的window_size个数据点
    if len(prices_df) > window_size:
        fig.update_xaxes(range=[prices_df.index[-window_size], prices_df.index[-1]])
    
    # 添加最新的趋势信号标记
    signal_markers = add_trend_signals(prices_df)
    for marker in signal_markers:
        fig.add_trace(marker, row=1, col=1)
    
    # 显示图表
    fig.show()

def add_trend_signals(prices_df: pd.DataFrame) -> list:
    """
    在图表上添加趋势信号标记
    
    Args:
        prices_df: 包含OHLCV数据的DataFrame
        
    Returns:
        list: 包含信号标记的图形对象列表
    """
    signals = calculate_trend_signals(prices_df)
    
    # 获取最新的信号
    latest_signal = signals['signal']
    confidence = signals['confidence']
    
    # 创建信号标记
    signal_markers = []
    if latest_signal == 'bullish':
        signal_markers.append(
            go.Scatter(
                x=[prices_df.index[-1]],
                y=[prices_df['low'].iloc[-1]],
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=15, color='green'),
                text=[f'买入 ({confidence:.0f}%)'],
                textposition='bottom center',
                showlegend=False
            )
        )
    elif latest_signal == 'bearish':
        signal_markers.append(
            go.Scatter(
                x=[prices_df.index[-1]],
                y=[prices_df['high'].iloc[-1]],
                mode='markers+text',
                marker=dict(symbol='triangle-down', size=15, color='red'),
                text=[f'卖出 ({confidence:.0f}%)'],
                textposition='top center',
                showlegend=False
            )
        )
    
    return signal_markers 