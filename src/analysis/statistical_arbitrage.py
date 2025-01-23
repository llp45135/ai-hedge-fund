import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# 导入共享的配置
from analysis.momentum_analysis import WINDOW_CONFIG, check_data_quality
from analysis.volatility_analysis import calculate_linear_trend

def calculate_stat_arb_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    统计套利策略
    
    核心思想：
    1. 价格分布特征分析
    2. 均值回归特性检验
    3. 市场效率性分析
    4. 异常行为检测
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
    
    # 计算对数收益率
    log_returns = np.log(prices_df["close"] / prices_df["close"].shift(1))
    metrics = {}
    
    # 1. 基础统计特征
    if "short_term" in quality["available_windows"]:
        window = WINDOW_CONFIG["windows"]["short_term"]
        # 短期统计特征
        metrics.update(calculate_distribution_metrics(log_returns, window, "short_term"))
    
    # 2. 中期统计特征
    if "medium_term" in quality["available_windows"]:
        window = WINDOW_CONFIG["windows"]["medium_term"]
        # 中期统计特征
        metrics.update(calculate_distribution_metrics(log_returns, window, "medium_term"))
        
        # 计算Hurst指数（需要足够的数据点）
        metrics["hurst_exponent"] = float(calculate_hurst_exponent(prices_df["close"]))
        
        # 计算自相关性
        metrics["autocorrelation"] = float(calculate_autocorrelation(log_returns, window))
    
    # 3. 长期统计特征
    if "long_term" in quality["available_windows"]:
        window = WINDOW_CONFIG["windows"]["long_term"]
        # 长期统计特征
        metrics.update(calculate_distribution_metrics(log_returns, window, "long_term"))
        
        # 计算长期趋势强度
        metrics["trend_strength"] = float(calculate_trend_strength(prices_df["close"], window))
    
    # 4. 价格效率性分析
    metrics["price_efficiency"] = float(calculate_price_efficiency(prices_df["close"]))
    
    # 根据可用指标动态计算信号
    signal, confidence = calculate_stat_arb_signal(metrics)
    
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": metrics,
        "data_quality": quality
    }

def calculate_distribution_metrics(returns: pd.Series, window: int, prefix: str) -> Dict[str, float]:
    """计算收益率分布的统计特征"""
    metrics = {}
    
    # 基础统计量
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    
    # 标准化收益率（用于计算高阶矩）
    standardized_returns = (returns - rolling_mean) / rolling_std
    
    # 计算统计矩
    metrics[f"{prefix}_skewness"] = float(standardized_returns.rolling(window).skew().iloc[-1])
    metrics[f"{prefix}_kurtosis"] = float(standardized_returns.rolling(window).kurt().iloc[-1])
    
    # 计算分位数
    for q in [0.05, 0.25, 0.75, 0.95]:
        metrics[f"{prefix}_quantile_{int(q*100)}"] = float(returns.rolling(window).quantile(q).iloc[-1])
    
    return metrics

def calculate_autocorrelation(returns: pd.Series, lag: int) -> float:
    """计算收益率的自相关性"""
    return returns.autocorr(lag)

def calculate_price_efficiency(prices: pd.Series) -> float:
    """
    计算价格效率比率
    价格效率 = 直线距离 / 实际路径
    效率值接近1表示趋势性强，接近0表示波动性强
    """
    direct_distance = abs(prices.iloc[-1] - prices.iloc[0])
    path_distance = abs(prices.diff()).sum()
    return direct_distance / path_distance if path_distance != 0 else 1.0

def calculate_trend_strength(prices: pd.Series, window: int) -> float:
    """
    计算趋势强度
    使用线性回归R方值来衡量趋势强度
    """
    x = np.arange(len(prices[-window:]))
    y = prices[-window:].values
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return r_squared

def calculate_stat_arb_signal(metrics: Dict[str, float]) -> Tuple[str, float]:
    """
    基于统计特征综合计算交易信号
    
    信号逻辑：
    1. 均值回归信号：基于Hurst指数和自相关性
    2. 分布异常信号：基于偏度和峰度
    3. 效率信号：基于价格效率性
    4. 趋势信号：基于趋势强度
    """
    signal = "neutral"
    confidence = 0.5
    
    # 1. 评估均值回归特性
    if "hurst_exponent" in metrics and "autocorrelation" in metrics:
        hurst = metrics["hurst_exponent"]
        autocorr = metrics["autocorrelation"]
        
        # Hurst指数显示强均值回归特性
        if hurst < 0.4 and abs(autocorr) > 0.2:
            mean_reversion_signal = True
            mean_reversion_strength = (0.5 - hurst) * 2
        else:
            mean_reversion_signal = False
            mean_reversion_strength = 0
    else:
        mean_reversion_signal = False
        mean_reversion_strength = 0
    
    # 2. 评估分布异常
    if "medium_term_skewness" in metrics and "medium_term_kurtosis" in metrics:
        skew = metrics["medium_term_skewness"]
        kurt = metrics["medium_term_kurtosis"]
        
        # 显著的分布偏离
        if abs(skew) > 1 or abs(kurt) > 4:
            distribution_signal = True
            # 根据偏度方向确定信号
            distribution_direction = 1 if skew > 0 else -1
            distribution_strength = min((abs(skew) + abs(kurt)/4) / 4, 1.0)
        else:
            distribution_signal = False
            distribution_direction = 0
            distribution_strength = 0
    else:
        distribution_signal = False
        distribution_direction = 0
        distribution_strength = 0
    
    # 3. 评估价格效率性
    efficiency = metrics["price_efficiency"]
    efficiency_signal = efficiency < 0.3  # 低效率表示可能存在套利机会
    
    # 4. 综合信号
    if mean_reversion_signal and distribution_signal:
        # 均值回归和分布异常共同确认
        signal = "bullish" if distribution_direction < 0 else "bearish"
        confidence = min(mean_reversion_strength * distribution_strength * 1.2, 1.0)
        
        # 如果效率性也确认，提高置信度
        if efficiency_signal:
            confidence = min(confidence * 1.2, 1.0)
    
    elif mean_reversion_signal:
        # 仅有均值回归信号
        recent_returns = metrics.get("short_term_quantile_75", 0) - metrics.get("short_term_quantile_25", 0)
        signal = "bullish" if recent_returns < 0 else "bearish"
        confidence = mean_reversion_strength * 0.8
    
    elif distribution_signal and efficiency_signal:
        # 分布异常和低效率共同确认
        signal = "bullish" if distribution_direction < 0 else "bearish"
        confidence = distribution_strength * 0.7
    
    return signal, confidence

def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent to determine long-term memory of time series
    H < 0.5: Mean reverting series
    H = 0.5: Random walk
    H > 0.5: Trending series
    """
    lags = range(2, max_lag)
    # Add small epsilon to avoid log(0)
    tau = [max(1e-8, np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag])))) for lag in lags]
    
    # Return the Hurst exponent from linear fit
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]  # Hurst exponent is the slope
    except (ValueError, RuntimeWarning):
        # Return 0.5 (random walk) if calculation fails
        return 0.5 