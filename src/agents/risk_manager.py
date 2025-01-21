import math
from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
import json
import ast
from tools.yfinance_api import get_prices, prices_to_df
from utils.progress import progress


##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Controls position sizing based on real-world risk factors."""
    data = state.get("data", {})
    portfolio = data.get("portfolio", {})
    tickers = data.get("tickers", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    
    # 初始化风险管理结果
    risk_analysis = {}
    
    # 处理每个股票
    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing risk factors")
        try:
            # 获取历史价格数据
            prices = get_prices(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            prices_df = prices_to_df(prices)
            
            # 获取当前价格和成交量数据
            current_price = prices_df["close"].iloc[-1]
            avg_daily_volume = prices_df["volume"].mean()
            daily_dollar_volume = avg_daily_volume * current_price
            
            # 计算投资组合价值
            position_value = portfolio["positions"].get(ticker, 0) * current_price
            total_portfolio_value = portfolio["cash"] + sum(
                portfolio["positions"].get(t, 0) * prices_df["close"].iloc[-1]
                for t in portfolio["positions"]
            )
            
            # 1. 流动性限制 - 不超过日均成交量的10%
            liquidity_limit = daily_dollar_volume * 0.10
            
            # 2. 头寸规模限制 - 单个股票不超过投资组合的20%
            position_limit = total_portfolio_value * 0.20
            
            # 3. 剩余可用头寸
            current_position_value = portfolio["positions"].get(ticker, 0) * current_price
            remaining_position_limit = max(0, position_limit - current_position_value)
            
            # 最终头寸限制是所有限制中的最小值
            max_position_size = min(liquidity_limit, remaining_position_limit)
            
            # 计算风险信号
            # 如果剩余可用头寸很大（>50%），给出买入信号
            # 如果剩余可用头寸很小（<10%），给出卖出信号
            position_usage = current_position_value / position_limit if position_limit > 0 else 1
            
            if position_usage < 0.5:  # 使用率低于50%
                signal = "BUY"
                confidence = (0.5 - position_usage) * 2  # 头寸越小，信心越大
            elif position_usage > 0.9:  # 使用率超过90%
                signal = "SELL"
                confidence = min((position_usage - 0.9) * 10, 1)  # 头寸越大，信心越大
            else:
                signal = "HOLD"
                confidence = 0.5
            
            # 生成分析报告
            risk_analysis[ticker] = {
                "signal": signal,
                "confidence": float(confidence),
                "current_price": float(current_price),
                "max_position_size": float(max_position_size),
                "remaining_position_limit": float(remaining_position_limit),
                "current_position_value": float(current_position_value),
                "reasoning": {
                    "liquidity_analysis": f"Daily volume: ${daily_dollar_volume:,.2f}, Limit: ${liquidity_limit:,.2f}",
                    "position_analysis": f"Portfolio value: ${total_portfolio_value:,.2f}, Position limit: ${position_limit:,.2f}",
                    "current_status": f"Current position: ${current_position_value:,.2f}, Remaining limit: ${remaining_position_limit:,.2f}",
                    "signal_explanation": f"Position usage: {position_usage:.1%}, Signal: {signal}, Confidence: {confidence:.2f}"
                }
            }
            
        except Exception as e:
            print(f"Error analyzing risk for {ticker}: {str(e)}")
            risk_analysis[ticker] = {
                "signal": "HOLD",
                "confidence": 0.0,
                "error": str(e),
                "max_position_size": 0,
                "remaining_position_limit": 0,
                "current_position_value": 0
            }
        
        progress.update_status("risk_management_agent", ticker, "Done")
    
    # 创建风险管理消息
    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name="risk_management_agent",
    )
    
    # 如果需要显示推理过程
    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(risk_analysis, "Risk Management Agent")
    
    # 更新分析信号
    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis
    
    return {
        "messages": state["messages"] + [message],
        "data": data,
    }
