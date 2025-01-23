"""
技术分析可视化模块

该模块提供了一系列用于可视化技术分析指标的函数，
包括趋势分析、成交量分析、动量指标等的可视化。
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from analysis.trend_analysis import (
    calculate_volume_ema,
    calculate_volume_trend,
    calculate_price_volume_sync,
    calculate_support_resistance,
    calculate_price_position,
    calculate_trend_lines,
    analyze_trend_lines,
    calculate_volume_support,
    analyze_breakout,
    combine_trend_signals,
    calculate_adx,
    calculate_volume_rsi,
)

from agents.technicals import (
    calculate_ema,
    calculate_trend_signals,
    calculate_mean_reversion_signals,
    calculate_momentum_signals,
    calculate_volatility_signals,
    calculate_stat_arb_signals,
    weighted_signal_combination,
    WINDOW_CONFIG
)

def visualize_trend_signals(
    prices_df: pd.DataFrame, 
    window_size: int = 200,
    show_trend: bool = True,
    show_mean_reversion: bool = True,
    show_momentum: bool = True,
    show_volatility: bool = True,
    show_combined: bool = True  # 添加综合信号显示控制
) -> None:
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
        show_trend: 是否显示趋势信号，默认True
        show_mean_reversion: 是否显示均值回归信号，默认True
        show_momentum: 是否显示动量信号，默认True
        show_volatility: 是否显示波动率信号，默认True
        show_combined: 是否显示综合信号，默认True
        
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
    
    # 添加趋势信号标记
    if show_trend:
        trend_signals_markers = add_trend_signals(prices_df)
        print(f"处理的数据点数: {len(prices_df)}")
        print(f"生成的trend_signals信号数量: {len(trend_signals_markers)}")
        for marker in trend_signals_markers:
            fig.add_trace(marker, row=1, col=1)
    
    # 添加均值回归信号标记
    if show_mean_reversion:
        mean_reversion_markers = add_mean_reversion_signals(prices_df)
        print(f"生成的mean_reversion_signals信号数量: {len(mean_reversion_markers)}")
        for marker in mean_reversion_markers:
            fig.add_trace(marker, row=1, col=1)
    
    # 添加动量信号标记
    if show_momentum:
        momentum_markers = add_momentum_signals(prices_df)
        print(f"生成的momentum_signals信号数量: {len(momentum_markers)}")
        for marker in momentum_markers:
            fig.add_trace(marker, row=1, col=1)
    
    # 添加波动率信号标记
    if show_volatility:
        volatility_markers = add_volatility_signals(prices_df)
        print(f"生成的volatility_signals信号数量: {len(volatility_markers)}")
        for marker in volatility_markers:
            fig.add_trace(marker, row=1, col=1)
    
    # 添加综合信号标记
    if show_combined:
        combined_markers = add_combined_signals(prices_df)
        print(f"生成的combined_signals信号数量: {len(combined_markers)}")
        for marker in combined_markers:
            fig.add_trace(marker, row=1, col=1)
    
    # 显示图表
    fig.show()

def add_trend_signals(prices_df: pd.DataFrame) -> list:
    """
    在图表上添加所有历史趋势信号标记
    """
    signal_markers = []
    min_required_days = WINDOW_CONFIG["min_required_days"]
    
    # 预处理数据
    df, is_valid = preprocess_data(prices_df, min_required_days)
    if not is_valid:
        return signal_markers
    
    # 确保从有足够历史数据的点开始计算
    for i in range(min_required_days, len(df)):
        # 使用截至当前的数据计算信号
        current_df = df.iloc[:i+1].copy()
        
        # 检查当前数据段的长度
        if len(current_df) <= min_required_days:
            continue
            
        try:
            with np.errstate(all='ignore'):  # 忽略所有数值计算警告
                signals = calculate_trend_signals(current_df)
                
                # 只在信号发生变化时添加标记
                if i > min_required_days:
                    prev_df = df.iloc[:i].copy()
                    prev_signals = calculate_trend_signals(prev_df)
                    if signals['signal'] == prev_signals['signal']:
                        continue
                
                # 添加信号标记
                if signals['signal'] == 'bullish':
                    signal_markers.append(
                        go.Scatter(
                            x=[current_df.index[-1]],
                            y=[current_df['low'].iloc[-1]],
                            mode='markers+text',
                            marker=dict(
                                symbol='triangle-up',
                                size=15,
                                color='green'
                            ),
                            text=[f'趋势买入 ({signals["confidence"]:.0f}%)'],
                            textposition='bottom center',
                            name='买入信号',
                            showlegend=False
                        )
                    )
                elif signals['signal'] == 'bearish':
                    signal_markers.append(
                        go.Scatter(
                            x=[current_df.index[-1]],
                            y=[current_df['high'].iloc[-1]],
                            mode='markers+text',
                            marker=dict(
                                symbol='triangle-down',
                                size=15,
                                color='red'
                            ),
                            text=[f'趋势卖出 ({signals["confidence"]:.0f}%)'],
                            textposition='top center',
                            name='卖出信号',
                            showlegend=False
                        )
                    )
                
                if i % 50 == 0:
                    print(f"处理进度: {i}/{len(df)}, 当前信号: {signals['signal']}")
                    
        except Exception as e:
            if i % 50 == 0:  # 减少错误信息的输出频率
                print(f"计算信号时出错 (i={i}): {e}")
            continue
    
    print(f"总共生成信号数量: {len(signal_markers)}")
    return signal_markers

def add_mean_reversion_signals(prices_df: pd.DataFrame) -> list:
    """
    在图表上添加所有历史均值回归信号标记
    
    Args:
        prices_df: 包含OHLCV数据的DataFrame
        
    Returns:
        list: 包含所有信号标记的图形对象列表
    """
    signal_markers = []
    min_required_days = WINDOW_CONFIG["min_required_days"]
    
    # 确保从有足够历史数据的点开始计算
    for i in range(min_required_days, len(prices_df)):
        # 使用截至当前的数据计算信号，创建副本而不是视图
        current_df = prices_df.iloc[:i+1].copy()
        try:
            signals = calculate_mean_reversion_signals(current_df)
            
            # 只在信号发生变化时添加标记
            if i > min_required_days:  # 确保有前一个信号可比较
                prev_df = prices_df.iloc[:i].copy()
                prev_signals = calculate_mean_reversion_signals(prev_df)
                if signals['signal'] == prev_signals['signal']:
                    continue
            
            # 添加信号标记
            if signals['signal'] == 'bullish':
                signal_markers.append(
                    go.Scatter(
                        x=[current_df.index[-1]],
                        y=[current_df['low'].iloc[-1]],
                        mode='markers+text',
                        marker=dict(
                            symbol='circle',  # 使用圆形标记区分于趋势信号
                            size=12,
                            color='lightgreen'  # 使用较浅的绿色
                        ),
                        text=[f'回归买入 ({signals["confidence"]:.0f}%)'],
                        textposition='bottom center',
                        name='均值回归买入信号',
                        showlegend=False
                    )
                )
            elif signals['signal'] == 'bearish':
                signal_markers.append(
                    go.Scatter(
                        x=[current_df.index[-1]],
                        y=[current_df['high'].iloc[-1]],
                        mode='markers+text',
                        marker=dict(
                            symbol='circle',  # 使用圆形标记区分于趋势信号
                            size=12,
                            color='pink'  # 使用较浅的红色
                        ),
                        text=[f'回归卖出 ({signals["confidence"]:.0f}%)'],
                        textposition='top center',
                        name='均值回归卖出信号',
                        showlegend=False
                    )
                )
            
            if i % 50 == 0:  # 每50个点打印一次进度
                print(f"均值回归信号处理进度: {i}/{len(prices_df)}, 当前信号: {signals['signal']}")
                
        except Exception as e:
            print(f"计算均值回归信号时出错 (i={i}): {e}")
            continue
    
    print(f"总共生成均值回归信号数量: {len(signal_markers)}")
    return signal_markers

def add_momentum_signals(prices_df: pd.DataFrame) -> list:
    """
    在图表上添加所有历史动量信号标记
    
    Args:
        prices_df: 包含OHLCV数据的DataFrame
        
    Returns:
        list: 包含所有信号标记的图形对象列表
    """
    signal_markers = []
    min_required_days = WINDOW_CONFIG["min_required_days"]
    
    # 确保从有足够历史数据的点开始计算
    for i in range(min_required_days, len(prices_df)):
        # 使用截至当前的数据计算信号，创建副本而不是视图
        current_df = prices_df.iloc[:i+1].copy()
        try:
            signals = calculate_momentum_signals(current_df)
            
            # 只在信号发生变化时添加标记
            if i > min_required_days:  # 确保有前一个信号可比较
                prev_df = prices_df.iloc[:i].copy()
                prev_signals = calculate_momentum_signals(prev_df)
                if signals['signal'] == prev_signals['signal']:
                    continue
            
            # 添加信号标记
            if signals['signal'] == 'bullish':
                signal_markers.append(
                    go.Scatter(
                        x=[current_df.index[-1]],
                        y=[current_df['low'].iloc[-1]],
                        mode='markers+text',
                        marker=dict(
                            symbol='diamond',  # 使用菱形标记区分于其他信号
                            size=12,
                            color='lime'  # 使用鲜艳的绿色
                        ),
                        text=[f'动量买入 ({signals["confidence"]:.0f}%)'],
                        textposition='bottom center',
                        name='动量买入信号',
                        showlegend=False
                    )
                )
            elif signals['signal'] == 'bearish':
                signal_markers.append(
                    go.Scatter(
                        x=[current_df.index[-1]],
                        y=[current_df['high'].iloc[-1]],
                        mode='markers+text',
                        marker=dict(
                            symbol='diamond',  # 使用菱形标记区分于其他信号
                            size=12,
                            color='crimson'  # 使用鲜艳的红色
                        ),
                        text=[f'动量卖出 ({signals["confidence"]:.0f}%)'],
                        textposition='top center',
                        name='动量卖出信号',
                        showlegend=False
                    )
                )
            
            if i % 50 == 0:  # 每50个点打印一次进度
                print(f"动量信号处理进度: {i}/{len(prices_df)}, 当前信号: {signals['signal']}")
                
        except Exception as e:
            print(f"计算动量信号时出错 (i={i}): {e}")
            continue
    
    print(f"总共生成动量信号数量: {len(signal_markers)}")
    return signal_markers

def add_volatility_signals(prices_df: pd.DataFrame) -> list:
    """
    在图表上添加所有历史波动率信号标记
    
    Args:
        prices_df: 包含OHLCV数据的DataFrame
        
    Returns:
        list: 包含所有信号标记的图形对象列表
    """
    signal_markers = []
    min_required_days = WINDOW_CONFIG["min_required_days"]
    
    # 确保从有足够历史数据的点开始计算
    for i in range(min_required_days, len(prices_df)):
        # 使用截至当前的数据计算信号，创建副本而不是视图
        current_df = prices_df.iloc[:i+1].copy()
        try:
            signals = calculate_volatility_signals(current_df)
            
            # 只在信号发生变化时添加标记
            if i > min_required_days:  # 确保有前一个信号可比较
                prev_df = prices_df.iloc[:i].copy()
                prev_signals = calculate_volatility_signals(prev_df)
                if signals['signal'] == prev_signals['signal']:
                    continue
            
            # 添加信号标记
            if signals['signal'] == 'bullish':
                signal_markers.append(
                    go.Scatter(
                        x=[current_df.index[-1]],
                        y=[current_df['low'].iloc[-1]],
                        mode='markers+text',
                        marker=dict(
                            symbol='star',  # 使用星形标记区分于其他信号
                            size=15,
                            color='yellowgreen'  # 使用黄绿色
                        ),
                        text=[f'波动买入 ({signals["confidence"]:.0f}%)'],
                        textposition='bottom center',
                        name='波动率买入信号',
                        showlegend=False
                    )
                )
            elif signals['signal'] == 'bearish':
                signal_markers.append(
                    go.Scatter(
                        x=[current_df.index[-1]],
                        y=[current_df['high'].iloc[-1]],
                        mode='markers+text',
                        marker=dict(
                            symbol='star',  # 使用星形标记区分于其他信号
                            size=15,
                            color='coral'  # 使用珊瑚红色
                        ),
                        text=[f'波动卖出 ({signals["confidence"]:.0f}%)'],
                        textposition='top center',
                        name='波动率卖出信号',
                        showlegend=False
                    )
                )
            
            if i % 50 == 0:  # 每50个点打印一次进度
                print(f"波动率信号处理进度: {i}/{len(prices_df)}, 当前信号: {signals['signal']}")
                
        except Exception as e:
            print(f"计算波动率信号时出错 (i={i}): {e}")
            continue
    
    print(f"总共生成波动率信号数量: {len(signal_markers)}")
    return signal_markers

def add_combined_signals(prices_df: pd.DataFrame) -> list:
    """
    在图表上添加综合策略信号标记
    
    Args:
        prices_df: 包含OHLCV数据的DataFrame
        
    Returns:
        list: 包含所有信号标记的图形对象列表
    """
    signal_markers = []
    min_required_days = WINDOW_CONFIG["min_required_days"]
    
    # 定义策略权重
    strategy_weights = {
        "trend": 0.25,
        "mean_reversion": 0.20,
        "momentum": 0.25,
        "volatility": 0.15,
        "stat_arb": 0.15,
    }
    
    try:
        # 预处理：创建副本并进行数据清理
        df = prices_df.copy()
        
        # 1. 检查是否有无穷值
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 2. 向前填充，然后向后填充
        df = df.ffill().bfill()
        
        # 3. 检查是否仍有缺失值
        if df.isnull().any().any():
            print("警告: 数据中存在无法填充的缺失值")
            return signal_markers
        
        # 4. 确保数据量足够
        if len(df) < min_required_days * 2:  # 确保有足够的数据进行计算
            print(f"警告: 数据点数量({len(df)})不足以进行可靠的信号计算")
            return signal_markers
        
        # 5. 检查数据的有效性
        if (df['high'] < df['low']).any() or (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
            print("警告: 数据中存在无效的OHLC值")
            return signal_markers
        
        # 6. 检查成交量的有效性
        if (df['volume'] <= 0).any():
            print("警告: 数据中存在无效的成交量值")
            return signal_markers
        
        # 确保从有足够历史数据的点开始计算
        for i in range(min_required_days, len(df)):
            # 使用截至当前的数据计算信号
            current_df = df.iloc[:i+1].copy()
            
            # 检查当前数据段的长度
            if len(current_df) <= min_required_days:
                continue
                
            try:
                with np.errstate(all='ignore'):  # 忽略所有数值计算警告
                    # 计算各个策略的信号，并进行错误处理
                    signals = {}
                    for name, calculator in [
                        ("trend", calculate_trend_signals),
                        ("mean_reversion", calculate_mean_reversion_signals),
                        ("momentum", calculate_momentum_signals),
                        ("volatility", calculate_volatility_signals),
                        ("stat_arb", calculate_stat_arb_signals)
                    ]:
                        try:
                            signals[name] = calculator(current_df)
                        except Exception as e:
                            if i % 50 == 0:  # 减少警告信息的输出频率
                                print(f"警告: 计算 {name} 信号时出错 (i={i}): {e}")
                            signals[name] = {"signal": "neutral", "confidence": 0.5}
                    
                    # 检查是否所有信号都计算成功
                    if len(signals) != 5:
                        continue
                    
                    # 组合信号
                    combined_signal = weighted_signal_combination(signals, strategy_weights)
                    
                    # 只在信号发生变化时添加标记
                    if i > min_required_days:
                        prev_df = df.iloc[:i].copy()
                        prev_signals = {}
                        
                        # 计算前一个时间点的信号
                        for name, calculator in [
                            ("trend", calculate_trend_signals),
                            ("mean_reversion", calculate_mean_reversion_signals),
                            ("momentum", calculate_momentum_signals),
                            ("volatility", calculate_volatility_signals),
                            ("stat_arb", calculate_stat_arb_signals)
                        ]:
                            try:
                                prev_signals[name] = calculator(prev_df)
                            except Exception:
                                prev_signals[name] = {"signal": "neutral", "confidence": 0.5}
                        
                        prev_combined = weighted_signal_combination(prev_signals, strategy_weights)
                        
                        if combined_signal["signal"] == prev_combined["signal"]:
                            continue
                    
                    # 添加信号标记
                    if combined_signal["signal"] == 'bullish':
                        signal_markers.append(
                            go.Scatter(
                                x=[current_df.index[-1]],
                                y=[current_df['low'].iloc[-1]],
                                mode='markers+text',
                                marker=dict(
                                    symbol='hexagon',
                                    size=18,
                                    color='mediumseagreen'
                                ),
                                text=[f'综合买入 ({combined_signal["confidence"]:.0f}%)'],
                                textposition='bottom center',
                                name='综合买入信号',
                                showlegend=False
                            )
                        )
                    elif combined_signal["signal"] == 'bearish':
                        signal_markers.append(
                            go.Scatter(
                                x=[current_df.index[-1]],
                                y=[current_df['high'].iloc[-1]],
                                mode='markers+text',
                                marker=dict(
                                    symbol='hexagon',
                                    size=18,
                                    color='indianred'
                                ),
                                text=[f'综合卖出 ({combined_signal["confidence"]:.0f}%)'],
                                textposition='top center',
                                name='综合卖出信号',
                                showlegend=False
                            )
                        )
                    
                    if i % 50 == 0:
                        print(f"综合信号处理进度: {i}/{len(df)}, 当前信号: {combined_signal['signal']}")
                        
            except Exception as e:
                if i % 50 == 0:  # 减少错误信息的输出频率
                    print(f"计算综合信号时出错 (i={i}): {e}")
                continue
                
    except Exception as e:
        print(f"数据预处理时出错: {e}")
        return signal_markers
    
    print(f"总共生成综合信号数量: {len(signal_markers)}")
    return signal_markers

def preprocess_data(prices_df: pd.DataFrame, min_required_days: int) -> tuple[pd.DataFrame, bool]:
    """
    预处理数据并进行有效性检查
    
    Args:
        prices_df: 原始数据DataFrame
        min_required_days: 所需的最小数据天数
        
    Returns:
        tuple: (处理后的DataFrame, 是否有效的布尔值)
    """
    try:
        # 创建副本并进行数据清理
        df = prices_df.copy()
        
        # 1. 检查是否有无穷值
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 2. 向前填充，然后向后填充
        df = df.ffill().bfill()
        
        # 3. 检查是否仍有缺失值
        if df.isnull().any().any():
            print("警告: 数据中存在无法填充的缺失值")
            return df, False
        
        # 4. 确保数据量足够
        if len(df) < min_required_days * 2:
            print(f"警告: 数据点数量({len(df)})不足以进行可靠的信号计算")
            return df, False
        
        # 5. 检查数据的有效性
        if (df['high'] < df['low']).any() or (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
            print("警告: 数据中存在无效的OHLC值")
            return df, False
        
        # 6. 检查成交量的有效性
        if (df['volume'] <= 0).any():
            print("警告: 数据中存在无效的成交量值")
            return df, False
            
        return df, True
        
    except Exception as e:
        print(f"数据预处理时出错: {e}")
        return prices_df, False 