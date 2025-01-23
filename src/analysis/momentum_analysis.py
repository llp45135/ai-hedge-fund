import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def calculate_momentum_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
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

# 配置常量
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