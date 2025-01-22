import math

from langchain_core.messages import HumanMessage

from graph.state import AgentState, show_agent_reasoning

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from tools.yfinance_api import get_prices, prices_to_df
from utils.progress import progress

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
    改进的趋势跟踪策略,增加成交量确认
    
    核心改进:
    1. 多周期成交量分析
    2. 成交量相对强度指标
    3. 价格-成交量协同性分析
    4. 动态调整趋势置信度
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
    
    # === 2. 成交量趋势分析 ===
    volume_trend = calculate_volume_trend(prices_df, volume_ema_5, volume_ema_20)
    
    # === 3. 价格-成交量协同性分析 ===
    price_volume_sync = calculate_price_volume_sync(prices_df, ema_8)
    
    # === 4. 成交量相对强度 ===
    volume_rsi = calculate_volume_rsi(prices_df)
    
    # === 5. 综合分析 ===
    trend_strength = adx["adx"].iloc[-1] / 100.0
    
    # 基础趋势信号
    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "bullish"
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5
    
    # 根据成交量特征调整置信度
    confidence = adjust_confidence_by_volume(
        signal,
        confidence,
        volume_trend,
        price_volume_sync,
        volume_rsi.iloc[-1]
    )

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": float(adx["adx"].iloc[-1]),
            "trend_strength": float(trend_strength),
            "volume_trend": float(volume_trend),
            "price_volume_sync": float(price_volume_sync),
            "volume_rsi": float(volume_rsi.iloc[-1])
        },
    }

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
    # 计算短期/长期成交量比值
    volume_ratio = volume_ema_5 / volume_ema_20
    
    # 计算成交量趋势强度(-1到1之间)
    trend_strength = (volume_ratio.iloc[-1] - 1) * 2
    
    # 限制在-1到1之间
    return max(min(trend_strength, 1), -1)

def calculate_price_volume_sync(df: pd.DataFrame, price_ema: pd.Series) -> float:
    """
    计算价格和成交量的协同性
    
    返回值:
    - 1: 完全协同
    - -1: 完全背离
    """
    # 计算价格和成交量的变化率
    price_change = price_ema.pct_change()
    volume_change = df["volume"].pct_change()
    
    # 计算最近N天的协同性
    window = 5
    sync_score = 0
    
    for i in range(-window, 0):
        if price_change.iloc[i] * volume_change.iloc[i] > 0:
            sync_score += 1
        else:
            sync_score -= 1
    
    return sync_score / window

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

def adjust_confidence_by_volume(
    signal: str,
    base_confidence: float,
    volume_trend: float,
    price_volume_sync: float,
    volume_rsi: float
) -> float:
    """
    根据成交量特征调整趋势信号的置信度
    
    Args:
        signal: 基础趋势信号
        base_confidence: 基础置信度
        volume_trend: 成交量趋势强度(-1到1)
        price_volume_sync: 价格-成交量协同性(-1到1)
        volume_rsi: 成交量RSI(0到100)
    """
    confidence = base_confidence
    
    # 1. 成交量趋势确认
    if signal == "bullish" and volume_trend > 0:
        confidence *= (1 + volume_trend * 0.2)  # 最多增加20%
    elif signal == "bearish" and volume_trend < 0:
        confidence *= (1 + abs(volume_trend) * 0.2)
    else:
        confidence *= 0.8  # 成交量不确认时降低置信度
    
    # 2. 价格-成交量协同性确认
    if price_volume_sync > 0:
        confidence *= (1 + price_volume_sync * 0.2)
    else:
        confidence *= (1 - abs(price_volume_sync) * 0.2)
    
    # 3. 成交量RSI确认
    if signal == "bullish" and volume_rsi > 50:
        confidence *= (1 + (volume_rsi - 50) / 250)  # 最多增加20%
    elif signal == "bearish" and volume_rsi < 50:
        confidence *= (1 + (50 - volume_rsi) / 250)
    else:
        confidence *= 0.9
    
    # 确保置信度在0到1之间
    return max(min(confidence, 1.0), 0.0)


def calculate_mean_reversion_signals(prices_df):
    """
    Mean reversion strategy using statistical measures and Bollinger Bands
    """
    # Calculate z-score of price relative to moving average
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50

    # Calculate Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)

    # Calculate RSI with multiple timeframes
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    # Mean reversion signals
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # Combine signals
    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = "bullish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = "bearish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": float(z_score.iloc[-1]),
            "price_vs_bb": float(price_vs_bb),
            "rsi_14": float(rsi_14.iloc[-1]),
            "rsi_28": float(rsi_28.iloc[-1]),
        },
    }


def calculate_momentum_signals(prices_df):
    """
    改进的多时间框架动量策略
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
    
    # 计算收益率
    returns = prices_df["close"].pct_change()
    metrics = {}
    
    # 根据可用窗口计算动量
    if "short_term" in quality["available_windows"]:
        metrics["momentum_1m"] = float(returns.rolling(WINDOW_CONFIG["windows"]["short_term"]).sum().iloc[-1])
        
    if "medium_term" in quality["available_windows"]:
        metrics["momentum_3m"] = float(returns.rolling(WINDOW_CONFIG["windows"]["medium_term"]).sum().iloc[-1])
        
    if "long_term" in quality["available_windows"]:
        metrics["momentum_6m"] = float(returns.rolling(WINDOW_CONFIG["windows"]["long_term"]).sum().iloc[-1])
    
    # 计算成交量动量（使用最短可用窗口）
    volume_ma = prices_df["volume"].rolling(WINDOW_CONFIG["windows"]["short_term"]).mean()
    metrics["volume_momentum"] = float(prices_df["volume"].iloc[-1] / volume_ma.iloc[-1])
    
    # 根据可用指标动态计算信号
    signal, confidence = calculate_momentum_signal(metrics)
    
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": metrics,
        "data_quality": quality
    }

def calculate_momentum_signal(metrics: Dict[str, float]) -> Tuple[str, float]:
    """
    根据可用的动量指标计算综合信号
    """
    # 初始化权重
    weights = {
        "momentum_1m": 0.4,
        "momentum_3m": 0.3,
        "momentum_6m": 0.3
    }
    
    # 调整权重基于可用指标
    available_weights = {k: v for k, v in weights.items() if k in metrics}
    if available_weights:
        # 重新归一化权重
        weight_sum = sum(available_weights.values())
        available_weights = {k: v/weight_sum for k, v in available_weights.items()}
    
    # 计算加权动量得分
    momentum_score = 0
    for indicator, weight in available_weights.items():
        momentum_score += metrics[indicator] * weight
    
    # 加入成交量确认
    volume_confirmation = metrics.get("volume_momentum", 1.0) > 1.0
    
    # 生成信号
    if momentum_score > 0.05 and volume_confirmation:
        signal = "bullish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal = "bearish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5
    
    return signal, confidence


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


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


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
