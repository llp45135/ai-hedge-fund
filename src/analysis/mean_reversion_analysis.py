import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def calculate_mean_reversion_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Mean reversion strategy using statistical measures and Bollinger Bands
    
    Returns:
        Dict containing:
        - signal: "bullish", "bearish", or "neutral"
        - confidence: signal confidence between 0 and 1
        - metrics: dict of calculated technical metrics
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

def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index
    
    Args:
        prices_df: DataFrame with OHLC data
        period: RSI calculation period
        
    Returns:
        pd.Series: RSI values
    """
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        prices_df: DataFrame with OHLC data
        window: Moving average window
        
    Returns:
        Tuple containing:
        - upper_band: Upper Bollinger Band
        - lower_band: Lower Bollinger Band
    """
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band 