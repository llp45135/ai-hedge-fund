import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy.stats import gaussian_kde

def analyze_trend(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    增强版趋势跟踪策略
    
    核心功能:
    1. 支撑/阻力位分析
    2. 关键价格水平分析
    3. 趋势线分析
    4. 成交量支撑度分析
    5. 突破确认机制
    """
    # 计算价格EMAs
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)

    # 计算成交量EMAs
    volume_ema_5 = calculate_volume_ema(prices_df, 5)
    volume_ema_20 = calculate_volume_ema(prices_df, 20)
    
    # 计算ADX
    adx = calculate_adx(prices_df, 14)
    
    # === 1. 价格趋势分析 ===
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55
    
    # === 2. 支撑/阻力位分析 ===
    support_resistance = calculate_support_resistance(prices_df)
    price_position = calculate_price_position(
        current_price=prices_df["close"].iloc[-1],
        support=support_resistance["support"],
        resistance=support_resistance["resistance"]
    )
    
    # === 3. 趋势线分析 ===
    trend_lines = calculate_trend_lines(prices_df)
    trend_line_signal = analyze_trend_lines(
        current_price=prices_df["close"].iloc[-1],
        trend_lines=trend_lines
    )
    
    # === 4. 成交量分析 ===
    volume_trend = calculate_volume_trend(prices_df, volume_ema_5, volume_ema_20)
    price_volume_sync = calculate_price_volume_sync(prices_df, ema_8)
    volume_support = calculate_volume_support(prices_df)
    
    # === 5. 突破分析 ===
    breakout_signal = analyze_breakout(
        prices_df,
        support_resistance,
        volume_support
    )
    
    # === 6. ADX分析 ===
    trend_strength = adx["adx"].iloc[-1] / 100.0
    
    # === 7. 综合分析 ===
    signal, confidence = combine_trend_signals(
        short_trend=short_trend.iloc[-1],
        medium_trend=medium_trend.iloc[-1],
        price_position=price_position,
        trend_line_signal=trend_line_signal,
        volume_trend=volume_trend,
        price_volume_sync=price_volume_sync,
        breakout_signal=breakout_signal,
        trend_strength=trend_strength
    )

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": float(adx["adx"].iloc[-1]),
            "trend_strength": float(trend_strength),
            "volume_trend": float(volume_trend),
            "price_volume_sync": float(price_volume_sync),
            "price_position": float(price_position),
            "support_level": float(support_resistance["support"]),
            "resistance_level": float(support_resistance["resistance"]),
            "breakout_strength": float(breakout_signal["strength"]),
            "volume_support": float(volume_support),
            "trend_line_score": float(trend_line_signal["score"])
        },
    }

# 把其他所有相关函数都移到这里
def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return df["close"].ewm(span=window, adjust=False).mean()

def calculate_volume_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """计算成交量的指数移动平均"""
    return df["volume"].ewm(span=window, adjust=False).mean()

def calculate_volume_trend(
    df: pd.DataFrame,
    volume_ema_5: pd.Series,
    volume_ema_20: pd.Series
) -> float:
    """
    计算成交量趋势强度
    
    返回值:
    - > 0: 成交量上升趋势
    - < 0: 成交量下降趋势
    - 绝对值表示趋势强度
    """
    volume_ratio = volume_ema_5 / volume_ema_20
    trend_strength = (volume_ratio.iloc[-1] - 1) * 2
    return max(min(trend_strength, 1), -1)

def calculate_price_volume_sync(df: pd.DataFrame, price_ema: pd.Series) -> float:
    """
    计算价格和成交量的协同性
    
    返回值:
    - 1: 完全协同
    - -1: 完全背离
    """
    price_change = price_ema.pct_change()
    volume_change = df["volume"].pct_change()
    window = 5
    sync_score = 0
    
    for i in range(-window, 0):
        if price_change.iloc[i] * volume_change.iloc[i] > 0:
            sync_score += 1
        else:
            sync_score -= 1
    
    return sync_score / window

def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
    """
    使用局部最高/最低点识别支撑位和阻力位
    """
    # 获取窗口内的高低点
    highs = df["high"].rolling(window=window, center=True).max()
    lows = df["low"].rolling(window=window, center=True).min()
    
    # 识别关键价格水平
    recent_highs = highs.tail(window)
    recent_lows = lows.tail(window)
    
    # 使用KDE识别价格密集区
    kde_lows = gaussian_kde(recent_lows.dropna())
    support_candidates = np.linspace(recent_lows.min(), recent_lows.max(), 50)
    support = support_candidates[np.argmax(kde_lows(support_candidates))]
    
    # 计算阻力位
    kde_highs = gaussian_kde(recent_highs.dropna())
    resistance_candidates = np.linspace(recent_highs.min(), recent_highs.max(), 50)
    resistance = resistance_candidates[np.argmax(kde_highs(resistance_candidates))]
    
    return {
        "support": support,
        "resistance": resistance
    }

def calculate_price_position(current_price: float, support: float, resistance: float) -> float:
    """
    计算当前价格在支撑/阻力区间的相对位置
    返回值在0-1之间，0表示在支撑位，1表示在阻力位
    """
    if resistance == support:
        return 0.5
    return (current_price - support) / (resistance - support)

def calculate_trend_lines(df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
    """
    计算趋势线
    返回上升和下降趋势线的参数
    """
    # 获取窗口数据
    recent_data = df.tail(window)
    
    # 计算上升趋势线
    highs = recent_data["high"].values
    x_highs = np.arange(len(highs))
    up_trend = np.polyfit(x_highs, highs, 1)
    
    # 计算下降趋势线
    lows = recent_data["low"].values
    x_lows = np.arange(len(lows))
    down_trend = np.polyfit(x_lows, lows, 1)
    
    return {
        "up_trend": up_trend,
        "down_trend": down_trend
    }

def analyze_trend_lines(current_price: float, trend_lines: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析价格相对于趋势线的位置
    """
    # 计算当前趋势线值
    x_current = len(trend_lines["up_trend"]) - 1
    up_trend_value = np.polyval(trend_lines["up_trend"], x_current)
    down_trend_value = np.polyval(trend_lines["down_trend"], x_current)
    
    # 计算价格相对于趋势线的位置
    up_trend_diff = current_price - up_trend_value
    down_trend_diff = current_price - down_trend_value
    
    # 计算趋势强度得分
    score = 0
    if up_trend_diff > 0 and down_trend_diff > 0:
        score = 1  # 强势上涨
    elif up_trend_diff < 0 and down_trend_diff < 0:
        score = -1  # 强势下跌
    else:
        # 在趋势线之间，根据相对位置计算得分
        score = (up_trend_diff + down_trend_diff) / (up_trend_value - down_trend_value)
    
    return {
        "score": score,
        "up_trend_diff": up_trend_diff,
        "down_trend_diff": down_trend_diff
    }

def calculate_volume_support(df: pd.DataFrame, window: int = 20) -> float:
    """
    计算成交量支撑度
    分析价格变动时的成交量确认程度
    """
    recent_data = df.tail(window)
    
    # 计算价格变动和对应的成交量
    price_changes = recent_data["close"].pct_change()
    volume_changes = recent_data["volume"].pct_change()
    
    # 计算价格上涨时的成交量支撑
    up_days = price_changes > 0
    up_volume_support = (volume_changes[up_days] > 0).mean()
    
    # 计算价格下跌时的成交量确认
    down_days = price_changes < 0
    down_volume_confirm = (volume_changes[down_days] > 0).mean()
    
    # 综合得分
    return up_volume_support - down_volume_confirm

def analyze_breakout(df: pd.DataFrame, support_resistance: Dict[str, float], volume_support: float) -> Dict[str, Any]:
    """
    分析价格突破
    考虑支撑/阻力位突破的有效性
    """
    current_price = df["close"].iloc[-1]
    current_volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].tail(20).mean()
    
    # 计算突破强度
    if current_price > support_resistance["resistance"]:
        breakout_type = "up"
        strength = (current_price - support_resistance["resistance"]) / support_resistance["resistance"]
    elif current_price < support_resistance["support"]:
        breakout_type = "down"
        strength = (support_resistance["support"] - current_price) / support_resistance["support"]
    else:
        breakout_type = "none"
        strength = 0
    
    # 成交量确认
    volume_confirm = current_volume > avg_volume * 1.5
    
    return {
        "type": breakout_type,
        "strength": strength,
        "volume_confirmed": volume_confirm
    }

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Average Directional Index (ADX)"""
    # Calculate True Range
    df = df.copy()
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    # Calculate Directional Movement
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]

    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    # Calculate ADX
    df["+di"] = 100 * (df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["-di"] = 100 * (df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]

def combine_trend_signals(
    short_trend: bool,
    medium_trend: bool,
    price_position: float,
    trend_line_signal: Dict[str, Any],
    volume_trend: float,
    price_volume_sync: float,
    breakout_signal: Dict[str, Any],
    trend_strength: float
) -> Tuple[str, float]:
    """
    综合各种趋势信号
    使用加权方法计算最终信号
    """
    # 初始化信号强度
    bullish_strength = 0
    bearish_strength = 0
    
    # 1. 趋势信号权重
    trend_weight = 0.3
    if short_trend and medium_trend:
        bullish_strength += trend_weight
    elif not short_trend and not medium_trend:
        bearish_strength += trend_weight
    
    # 2. 支撑/阻力位权重
    sr_weight = 0.2
    if price_position < 0.3:  # 接近支撑位
        bullish_strength += sr_weight * (1 - price_position)
    elif price_position > 0.7:  # 接近阻力位
        bearish_strength += sr_weight * price_position
    
    # 3. 趋势线权重
    trendline_weight = 0.15
    trendline_score = trend_line_signal["score"]
    if trendline_score > 0:
        bullish_strength += trendline_weight * trendline_score
    else:
        bearish_strength += trendline_weight * abs(trendline_score)
    
    # 4. 成交量权重
    volume_weight = 0.2
    if volume_trend > 0 and price_volume_sync > 0:
        bullish_strength += volume_weight * min(volume_trend, price_volume_sync)
    elif volume_trend < 0 and price_volume_sync < 0:
        bearish_strength += volume_weight * min(abs(volume_trend), abs(price_volume_sync))
    
    # 5. 突破权重
    breakout_weight = 0.15
    if breakout_signal["type"] == "up" and breakout_signal["volume_confirmed"]:
        bullish_strength += breakout_weight * breakout_signal["strength"]
    elif breakout_signal["type"] == "down" and breakout_signal["volume_confirmed"]:
        bearish_strength += breakout_weight * breakout_signal["strength"]
    
    # 计算最终信号
    net_strength = bullish_strength - bearish_strength
    
    # 根据趋势强度调整置信度
    confidence = abs(net_strength) * trend_strength
    
    # 确定信号方向
    if net_strength > 0.1:
        signal = "bullish"
    elif net_strength < -0.1:
        signal = "bearish"
    else:
        signal = "neutral"
        confidence = 0.5
    
    return signal, min(confidence, 1.0)

def calculate_volume_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算成交量的RSI"""
    volume_change = df["volume"].diff()
    
    # 分别计算成交量增加和减少
    gains = volume_change.copy()
    losses = volume_change.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # 计算RSI
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# ... 其他所有函数保持不变 ... 