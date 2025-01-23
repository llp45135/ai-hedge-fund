import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# 导入共享的配置
from analysis.momentum_analysis import WINDOW_CONFIG, check_data_quality

def calculate_volatility_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    波动率分析策略
    
    核心思想：
    1. 波动率具有均值回归特性
    2. 波动率变化往往领先于价格变化
    3. 极端波动率环境通常预示着市场转折
    """
    # 检查数据质量
    quality = check_data_quality(prices_df)
    if not quality["is_valid"]:
        return {
            "signal": "neutral",
            "confidence": 0.5,
            "metrics": {"warning": quality["warning"]},
            "data_quality": quality
        }
    
    # 计算对数收益率（更符合正态分布假设）
    log_returns = np.log(prices_df["close"] / prices_df["close"].shift(1))
    metrics = {}
    
    # 1. 基础波动率计算
    # 使用最短窗口计算基础波动率
    short_vol = log_returns.rolling(WINDOW_CONFIG["windows"]["short_term"]).std() * math.sqrt(252)
    metrics["current_volatility"] = float(short_vol.iloc[-1])
    
    # 2. 波动率趋势分析
    if "medium_term" in quality["available_windows"]:
        # 计算中期波动率
        medium_vol = log_returns.rolling(WINDOW_CONFIG["windows"]["medium_term"]).std() * math.sqrt(252)
        metrics["medium_term_volatility"] = float(medium_vol.iloc[-1])
        
        # 波动率变化率
        vol_change = (short_vol / medium_vol - 1) * 100
        metrics["volatility_change"] = float(vol_change.iloc[-1])
        
        # 波动率趋势（使用简单线性回归）
        vol_trend = calculate_linear_trend(short_vol.tail(WINDOW_CONFIG["windows"]["medium_term"]))
        metrics["volatility_trend"] = float(vol_trend)
    
    # 3. 波动率均值回归分析
    if "long_term" in quality["available_windows"]:
        # 长期波动率均值
        long_vol = log_returns.rolling(WINDOW_CONFIG["windows"]["long_term"]).std() * math.sqrt(252)
        metrics["long_term_volatility"] = float(long_vol.iloc[-1])
        
        # 计算当前波动率相对于长期均值的偏离程度
        vol_deviation = (short_vol - long_vol) / long_vol
        metrics["volatility_deviation"] = float(vol_deviation.iloc[-1])
        
        # 计算波动率的波动率（二阶波动率）
        vol_of_vol = short_vol.rolling(WINDOW_CONFIG["windows"]["medium_term"]).std() / short_vol.rolling(WINDOW_CONFIG["windows"]["medium_term"]).mean()
        metrics["volatility_of_volatility"] = float(vol_of_vol.iloc[-1])
    
    # 4. 计算真实波动率指标（考虑跳空）
    atr = calculate_atr(prices_df, WINDOW_CONFIG["windows"]["short_term"])
    metrics["atr_ratio"] = float(atr.iloc[-1] / prices_df["close"].iloc[-1])
    
    # 根据可用指标动态计算信号
    signal, confidence = calculate_volatility_signal(metrics)
    
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": metrics,
        "data_quality": quality
    }

def calculate_stat_arb_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """统计套利策略"""
    # ... 从 statistical_arbitrage.py 移动过来 ...

def calculate_linear_trend(series: pd.Series) -> float:
    """使用简单线性回归计算趋势斜率"""
    x = np.arange(len(series))
    y = series.values
    slope, _ = np.polyfit(x, y, 1)
    return slope

def calculate_distribution_metrics(returns: pd.Series, window: int, prefix: str) -> Dict[str, float]:
    """计算收益率分布的统计特征"""
    # ... 从 statistical_arbitrage.py 移动过来 ...

def calculate_autocorrelation(returns: pd.Series, lag: int) -> float:
    """计算收益率的自相关性"""
    # ... 从 statistical_arbitrage.py 移动过来 ...

def calculate_price_efficiency(prices: pd.Series) -> float:
    """计算价格效率比率"""
    # ... 从 statistical_arbitrage.py 移动过来 ...

def calculate_trend_strength(prices: pd.Series, window: int) -> float:
    """计算趋势强度"""
    # ... 从 statistical_arbitrage.py 移动过来 ...

def calculate_stat_arb_signal(metrics: Dict[str, float]) -> Tuple[str, float]:
    """基于统计特征综合计算交易信号"""
    # ... 从 statistical_arbitrage.py 移动过来 ...

def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """计算 Hurst 指数"""
    # ... 从 statistical_arbitrage.py 移动过来 ...

def calculate_volatility_signal(metrics: Dict[str, float]) -> Tuple[str, float]:
    """
    基于波动率指标综合计算交易信号
    
    信号逻辑：
    1. 基础信号：基于当前波动率水平
    2. 趋势信号：基于波动率变化趋势
    3. 均值回归信号：基于波动率偏离程度
    """
    signal = "neutral"
    confidence = 0.5
    
    # 1. 基础波动率评估
    current_vol = metrics["current_volatility"]
    
    # 2. 考虑波动率趋势（如果可用）
    if "volatility_trend" in metrics and "volatility_change" in metrics:
        vol_trend = metrics["volatility_trend"]
        vol_change = metrics["volatility_change"]
        
        # 波动率快速下降可能预示着市场即将上涨
        if vol_trend < 0 and vol_change < -10:
            signal = "bullish"
            confidence = min(abs(vol_change) / 20, 0.8)
        # 波动率快速上升可能预示着市场即将下跌
        elif vol_trend > 0 and vol_change > 10:
            signal = "bearish"
            confidence = min(abs(vol_change) / 20, 0.8)
    
    # 3. 考虑均值回归（如果可用）
    if "volatility_deviation" in metrics and "volatility_of_volatility" in metrics:
        vol_dev = metrics["volatility_deviation"]
        vol_of_vol = metrics["volatility_of_volatility"]
        
        # 极端偏离通常预示着反转
        if abs(vol_dev) > 2:  # 显著偏离
            # 如果波动率的波动率也很高，增加信号强度
            if vol_of_vol > 0.2:
                confidence = min(confidence * 1.2, 1.0)
            
            # 当前信号是中性时，生成新信号
            if signal == "neutral":
                signal = "bearish" if vol_dev > 0 else "bullish"
                confidence = min(abs(vol_dev) / 3, 0.8)
    
    # 4. 使用ATR确认（始终可用）
    atr_ratio = metrics["atr_ratio"]
    if atr_ratio > 0.03 and signal == "bearish":  # 高ATR确认看跌
        confidence = min(confidence * 1.1, 1.0)
    elif atr_ratio < 0.01 and signal == "bullish":  # 低ATR确认看涨
        confidence = min(confidence * 1.1, 1.0)
    
    return signal, confidence

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period).mean() 