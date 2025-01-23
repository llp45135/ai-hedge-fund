import math

from langchain_core.messages import HumanMessage

from graph.state import AgentState, show_agent_reasoning

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from tools.yfinance_api import get_prices, prices_to_df
from utils.progress import progress

from analysis.trend_analysis import analyze_trend
from analysis.mean_reversion_analysis import (
    calculate_mean_reversion_signals,
    calculate_rsi,
    calculate_bollinger_bands,
)
from analysis.momentum_analysis import (
    calculate_momentum_signals,
    WINDOW_CONFIG,
)

from analysis.volatility_analysis import (
    calculate_volatility_signals,
    calculate_stat_arb_signals
)

# 添加窗口配置
WINDOW_CONFIG = {
    "min_required_days": 21,  # 最小需要的数据天数
    "windows": {
        "short_term": 21,    # 1个月
        "medium_term": 63,   # 3个月
        "long_term": 126,    # 6个月
    }
}

def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    检查数据质量并返回可用的时间窗口
    
    Returns:
        Dict 包含:
        - is_valid: 数据是否满足最小要求
        - available_windows: 可用的时间窗口列表
        - data_points: 可用数据点数量
        - warning: 警告信息（如果有）
    """
    available_days = len(df)
    
    result = {
        "is_valid": available_days >= WINDOW_CONFIG["min_required_days"],
        "available_windows": [],
        "data_points": available_days,
        "warning": None
    }
    
    if not result["is_valid"]:
        result["warning"] = f"Insufficient data: {available_days} days available, {WINDOW_CONFIG['min_required_days']} required"
        return result
        
    # 检查每个时间窗口的可用性
    for window_name, window_size in WINDOW_CONFIG["windows"].items():
        if available_days >= window_size:
            result["available_windows"].append(window_name)
            
    return result


##### Technical Analyst #####
def technical_analyst_agent(state: AgentState):
    """
    Sophisticated technical analysis system that combines multiple trading strategies for multiple tickers:
    1. Trend Following
    2. Mean Reversion
    3. Momentum
    4. Volatility Analysis
    5. Statistical Arbitrage Signals
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize analysis for each ticker
    technical_analysis = {}

    for ticker in tickers:
        progress.update_status("technical_analyst_agent", ticker, "Analyzing price data")

        # Get the historical price data
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        if not prices:
            progress.update_status("technical_analyst_agent", ticker, "Failed: No price data found")
            continue

        # Convert prices to a DataFrame
        prices_df = prices_to_df(prices)
        
        # 打印数据长度信息
        print(f"\n=== Data Analysis for {ticker} ===")
        print(f"Total trading days: {len(prices_df)}")
        print(f"Date range: from {prices_df.index.min()} to {prices_df.index.max()}")
        print(f"Required days for analysis:")
        print(f"- Momentum 3M: 63 days")
        print(f"- Momentum 6M: 126 days")
        print(f"- Volatility: 63 days")
        print(f"- Statistical: 63 days")
        print("============================\n")

        progress.update_status("technical_analyst_agent", ticker, "Calculating trend signals")
        trend_signals = calculate_trend_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating mean reversion")
        mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating momentum")
        momentum_signals = calculate_momentum_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Analyzing volatility")
        volatility_signals = calculate_volatility_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Statistical analysis")
        stat_arb_signals = calculate_stat_arb_signals(prices_df)

        # Combine all signals using a weighted ensemble approach
        strategy_weights = {
            "trend": 0.25,
            "mean_reversion": 0.20,
            "momentum": 0.25,
            "volatility": 0.15,
            "stat_arb": 0.15,
        }

        progress.update_status("technical_analyst_agent", ticker, "Combining signals")
        combined_signal = weighted_signal_combination(
            {
                "trend": trend_signals,
                "mean_reversion": mean_reversion_signals,
                "momentum": momentum_signals,
                "volatility": volatility_signals,
                "stat_arb": stat_arb_signals,
            },
            strategy_weights,
        )

        # Generate detailed analysis report for this ticker
        technical_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "strategy_signals": {
                "trend_following": {
                    "signal": trend_signals["signal"],
                    "confidence": round(trend_signals["confidence"] * 100),
                    "metrics": normalize_pandas(trend_signals["metrics"]),
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": round(mean_reversion_signals["confidence"] * 100),
                    "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": round(momentum_signals["confidence"] * 100),
                    "metrics": normalize_pandas(momentum_signals["metrics"]),
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": round(volatility_signals["confidence"] * 100),
                    "metrics": normalize_pandas(volatility_signals["metrics"]),
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": round(stat_arb_signals["confidence"] * 100),
                    "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                },
            },
        }
        progress.update_status("technical_analyst_agent", ticker, "Done")

    # Create the technical analyst message
    message = HumanMessage(
        content=json.dumps(technical_analysis),
        name="technical_analyst_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(technical_analysis, "Technical Analyst")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def calculate_trend_signals(prices_df):
    """
    增强版趋势跟踪策略
    
    核心改进:
    1. 支撑/阻力位分析
    2. 关键价格水平分析
    3. 趋势线分析
    4. 成交量支撑度分析
    5. 突破确认机制
    """
    return analyze_trend(prices_df)


def calculate_volatility_signals(prices_df):
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

def calculate_linear_trend(series: pd.Series) -> float:
    """
    使用简单线性回归计算趋势斜率
    """
    x = np.arange(len(series))
    y = series.values
    slope, _ = np.polyfit(x, y, 1)
    return slope

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


def calculate_stat_arb_signals(prices_df):
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
    """
    计算收益率分布的统计特征
    """
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
    """
    计算收益率的自相关性
    """
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


def weighted_signal_combination(signals, weights):
    """
    Combines multiple trading signals using a weighted approach
    """
    # Convert signals to numeric values
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    weighted_sum = 0
    total_confidence = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        weight = weights[strategy]
        confidence = signal["confidence"]

        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence

    # Normalize the weighted sum
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    # Convert back to signal
    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {"signal": signal, "confidence": abs(final_score)}


def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        df: DataFrame with price data
        window: EMA period

    Returns:
        pd.Series: EMA values
    """
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with OHLC data
        period: Period for calculations

    Returns:
        DataFrame with ADX values
    """
    # Calculate True Range
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


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent to determine long-term memory of time series
    H < 0.5: Mean reverting series
    H = 0.5: Random walk
    H > 0.5: Trending series

    Args:
        price_series: Array-like price data
        max_lag: Maximum lag for R/S calculation

    Returns:
        float: Hurst exponent
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
    from scipy.stats import gaussian_kde
    
    # 计算支撑位
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

def analyze_breakout(
    df: pd.DataFrame,
    support_resistance: Dict[str, float],
    volume_support: float
) -> Dict[str, Any]:
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
