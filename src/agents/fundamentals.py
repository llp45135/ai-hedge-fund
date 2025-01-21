from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import json
from tools.yfinance_api import get_financial_metrics
import logging
from typing import Dict, Any, List
import numpy as np

def calculate_trend_signal(current: float, historical: List[float], threshold: float = 0.1) -> tuple[str, str]:
    """计算指标的趋势信号
    
    Args:
        current: 当前值
        historical: 历史值列表
        threshold: 变化阈值
        
    Returns:
        (signal_type, description): 信号类型和描述
    """
    if not historical or len(historical) < 2:
        return "", ""
        
    trend = (current - historical[-1]) / abs(historical[-1]) if historical[-1] != 0 else 0
    if abs(trend) < threshold:
        return "", ""
        
    if trend > 0:
        return "bullish", f"showing positive trend with {trend:.1%} growth"
    else:
        return "bearish", f"showing negative trend with {trend:.1%} decline"

##### Fundamental Agent #####
def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers", [])
    
    fundamental_analysis = {}
    
    for ticker in tickers:
        progress.update_status("fundamentals_agent", ticker, "Analyzing fundamentals")
        try:
            # 获取该股票的财务指标
            stock_info = get_financial_metrics(
                ticker=ticker,
                report_period=end_date,
            )[0]  # 获取最新的报告
            
            # 初始化信号评分系统
            bullish_signals = []
            bearish_signals = []
            signal_weights = []
            
            # 分析PE比率
            if stock_info.get('trailingPE') and stock_info.get('forwardPE'):
                pe_ratio = float(stock_info['trailingPE'])
                forward_pe = float(stock_info['forwardPE'])
                weight = 0.7  # PE是重要指标，给予较高权重
                
                # PE分析
                if pe_ratio < 15:
                    bullish_signals.append(f"PE ratio ({pe_ratio:.2f}) is attractively low")
                    signal_weights.append(weight)
                elif pe_ratio > 30:
                    bearish_signals.append(f"PE ratio ({pe_ratio:.2f}) is relatively high")
                    signal_weights.append(weight)
                
                # PE趋势分析
                if forward_pe < pe_ratio:
                    bullish_signals.append(f"Forward PE ({forward_pe:.2f}) suggests improving earnings")
                    signal_weights.append(weight * 0.5)
                elif forward_pe > pe_ratio * 1.2:  # 20%以上的恶化
                    bearish_signals.append(f"Forward PE ({forward_pe:.2f}) suggests deteriorating earnings")
                    signal_weights.append(weight * 0.5)
            
            # 分析PB比率
            if stock_info.get('priceToBook'):
                pb_ratio = float(stock_info['priceToBook'])
                weight = 0.6
                if pb_ratio < 1:
                    bullish_signals.append(f"Price-to-Book ratio ({pb_ratio:.2f}) indicates potential undervaluation")
                    signal_weights.append(weight)
                elif pb_ratio > 5:
                    bearish_signals.append(f"Price-to-Book ratio ({pb_ratio:.2f}) suggests high valuation")
                    signal_weights.append(weight)
            
            # 分析利润率和趋势
            if stock_info.get('profitMargins'):
                profit_margin = float(stock_info['profitMargins'])
                weight = 0.65
                if profit_margin > 0.2:  # 20%以上的利润率
                    bullish_signals.append(f"Strong profit margin at {profit_margin:.1%}")
                    signal_weights.append(weight)
                elif profit_margin < 0:  # 负利润率
                    bearish_signals.append(f"Negative profit margin at {profit_margin:.1%}")
                    signal_weights.append(weight)
                
                # 分析利润率趋势
                if stock_info.get('earningsGrowth'):
                    earnings_growth = float(stock_info['earningsGrowth'])
                    if earnings_growth > 0.1:  # 10%以上的增长
                        bullish_signals.append(f"Strong earnings growth at {earnings_growth:.1%}")
                        signal_weights.append(weight * 0.6)
                    elif earnings_growth < -0.1:  # 10%以上的下降
                        bearish_signals.append(f"Declining earnings at {earnings_growth:.1%}")
                        signal_weights.append(weight * 0.6)
            
            # 分析收入增长
            if stock_info.get('revenueGrowth'):
                revenue_growth = float(stock_info['revenueGrowth'])
                weight = 0.75
                if revenue_growth > 0.15:  # 15%以上的收入增长
                    bullish_signals.append(f"Strong revenue growth at {revenue_growth:.1%}")
                    signal_weights.append(weight)
                elif revenue_growth < 0:  # 负收入增长
                    bearish_signals.append(f"Negative revenue growth at {revenue_growth:.1%}")
                    signal_weights.append(weight)

            # 分析债务状况
            if stock_info.get('debtToEquity') and stock_info.get('currentRatio'):
                debt_to_equity = float(stock_info['debtToEquity'])
                current_ratio = float(stock_info['currentRatio'])
                weight = 0.55
                
                # 综合债务分析
                if debt_to_equity < 50 and current_ratio > 2:
                    bullish_signals.append(f"Strong financial health with low debt ({debt_to_equity:.1f}%) and high liquidity ({current_ratio:.2f})")
                    signal_weights.append(weight)
                elif debt_to_equity > 200 or current_ratio < 1:
                    bearish_signals.append(f"Financial stress with high debt ({debt_to_equity:.1f}%) or low liquidity ({current_ratio:.2f})")
                    signal_weights.append(weight)

            # 分析回报指标
            if stock_info.get('returnOnEquity') and stock_info.get('returnOnAssets'):
                roe = float(stock_info['returnOnEquity'])
                roa = float(stock_info['returnOnAssets'])
                weight = 0.6
                
                # 综合回报分析
                if roe > 0.15 and roa > 0.05:  # 15% ROE和5% ROA为优秀水平
                    bullish_signals.append(f"Strong returns with ROE at {roe:.1%} and ROA at {roa:.1%}")
                    signal_weights.append(weight)
                elif roe < 0.05 or roa < 0.02:  # 5% ROE和2% ROA为警戒水平
                    bearish_signals.append(f"Poor returns with ROE at {roe:.1%} and ROA at {roa:.1%}")
                    signal_weights.append(weight)
            
            # 计算综合信号
            total_bullish_weight = sum(signal_weights[:len(bullish_signals)])
            total_bearish_weight = sum(signal_weights[len(bullish_signals):])
            total_weight = total_bullish_weight + total_bearish_weight or 1.0  # 避免除以零
            
            # 确定最终信号
            if total_bullish_weight > total_bearish_weight:
                signal = "BULLISH"
                confidence = (total_bullish_weight / total_weight) * 100
                main_signals = bullish_signals
            elif total_bearish_weight > total_bullish_weight:
                signal = "BEARISH"
                confidence = (total_bearish_weight / total_weight) * 100
                main_signals = bearish_signals
            else:
                signal = "NEUTRAL"
                confidence = 50.0
                main_signals = bullish_signals + bearish_signals

            # 生成详细的分析报告
            fundamental_analysis[ticker] = {
                "signal": signal,
                "confidence": round(confidence, 2),
                "reasoning": {
                    "main_signals": main_signals,
                    "metrics": {
                        "Valuation": {
                            "PE": stock_info.get('trailingPE'),
                            "Forward_PE": stock_info.get('forwardPE'),
                            "PB": stock_info.get('priceToBook'),
                            "PS": stock_info.get('priceToSalesTrailing12Months')
                        },
                        "Profitability": {
                            "Profit_Margin": stock_info.get('profitMargins'),
                            "Operating_Margin": stock_info.get('operatingMargins'),
                            "ROE": stock_info.get('returnOnEquity'),
                            "ROA": stock_info.get('returnOnAssets')
                        },
                        "Growth": {
                            "Revenue_Growth": stock_info.get('revenueGrowth'),
                            "Earnings_Growth": stock_info.get('earningsGrowth')
                        },
                        "Financial_Health": {
                            "Debt_To_Equity": stock_info.get('debtToEquity'),
                            "Current_Ratio": stock_info.get('currentRatio'),
                            "Quick_Ratio": stock_info.get('quickRatio')
                        }
                    }
                }
            }
            
            logging.info(f"Successfully analyzed fundamentals for {ticker}")
            
        except Exception as e:
            logging.error(f"Error analyzing fundamentals for {ticker}: {str(e)}")
            fundamental_analysis[ticker] = {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": {
                    "error": str(e),
                    "main_signals": ["Analysis failed due to data unavailability"]
                }
            }
        
        progress.update_status("fundamentals_agent", ticker, "Done")
    
    # 创建基本面分析消息
    message = HumanMessage(
        content=json.dumps(fundamental_analysis, indent=2),
        name="fundamentals_agent",
    )
    
    # 如果需要显示推理过程
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")
    
    # 更新分析信号
    state["data"]["analyst_signals"]["fundamentals_agent"] = fundamental_analysis
    
    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
