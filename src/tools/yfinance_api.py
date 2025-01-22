import yfinance as yf
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta
from data.models import CompanyNews, InsiderTrade

def get_financial_metrics(
    ticker: str,
    report_period: str = None,
    end_date: str = None,
    period: str = 'ttm',
    limit: int = 1
) -> List[Dict[str, Any]]:
    """获取财务指标
    
    Args:
        ticker: 股票代码
        report_period: 报告期间（可选）
        end_date: 结束日期（可选，与 report_period 作用相同）
        period: 期间类型，默认为 'ttm'（过去12个月）
        limit: 返回的最大记录数
        
    Returns:
        List[Dict[str, Any]]: 财务指标列表
    """
    if not ticker:
        raise ValueError("Ticker cannot be None or empty")
        
    effective_period = report_period or end_date
    
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # 直接返回原始的yfinance数据，保持键名一致
    return [info]

# 添加字段映射字典
FIELD_MAPPINGS = {
    'free_cash_flow': ['Free Cash Flow'],
    'net_income': ['Net Income', 'Net Income From Continuing Operations'],
    'depreciation_and_amortization': ['Depreciation And Amortization', 'Depreciation Amortization Depletion'],
    'capital_expenditure': ['Capital Expenditure'],
    'working_capital': ['Working Capital']
}

def search_line_items(
    ticker: str,
    line_items: List[str],
    end_date: str | None = None,
    start_date: str | None = None,
    period: str = 'ttm',
    limit: int = 1
) -> List[Dict[str, Any]]:
    """获取财务数据中的特定项目
    
    Args:
        ticker: 股票代码
        line_items: 需要查询的项目列表
        end_date: 结束日期（可选）
        start_date: 开始日期（可选）
        period: 期间类型，默认为 'ttm'（过去12个月）
        limit: 返回的最大记录数
        
    Returns:
        List[Dict[str, Any]]: 财务数据项目列表
    """
    print(f"\n=== Financial Statements Debug for {ticker} ===")
    
    stock = yf.Ticker(ticker)
    
    # 获取所有财务报表
    cashflow = stock.cashflow
    balance = stock.balance_sheet
    income = stock.income_stmt
    
    print("\n--- Cash Flow Statement ---")
    if not cashflow.empty:
        print("\nAnnual Cash Flow Items:")
        print(cashflow.index.tolist())
        print("\nSample Data:")
        print(cashflow.head())
    else:
        print("No annual cash flow data available")
        
    print("\n--- Balance Sheet ---")
    if not balance.empty:
        print("\nAnnual Balance Sheet Items:")
        print(balance.index.tolist())
        print("\nSample Data:")
        print(balance.head())
    else:
        print("No annual balance sheet data available")
    
    print("\n--- Income Statement ---")
    if not income.empty:
        print("\nAnnual Income Statement Items:")
        print(income.index.tolist())
        print("\nSample Data:")
        print(income.head())
    else:
        print("No annual income statement data available")
    
    # 如果提供了日期范围，进行日期筛选
    if end_date:
        end_date = pd.to_datetime(end_date)
        cashflow = cashflow.loc[:, cashflow.columns <= end_date]
        balance = balance.loc[:, balance.columns <= end_date]
        if not income.empty:
            income = income.loc[:, income.columns <= end_date]
        
    if start_date:
        start_date = pd.to_datetime(start_date)
        cashflow = cashflow.loc[:, cashflow.columns >= start_date]
        balance = balance.loc[:, balance.columns >= start_date]
        if not income.empty:
            income = income.loc[:, income.columns >= start_date]
    
    # 确保至少有一条数据
    if cashflow.empty and balance.empty and income.empty:
        print("\nNo financial data available after date filtering")
        return []
        
    results = []
    print("\n--- Searching for requested line items ---")
    print(f"Requested items: {line_items}")
    
    # 获取所有可用的日期
    all_dates = set()
    if not cashflow.empty:
        all_dates.update(cashflow.columns)
    if not balance.empty:
        all_dates.update(balance.columns)
    if not income.empty:
        all_dates.update(income.columns)
    
    # 按日期排序（降序）
    all_dates = sorted(list(all_dates), reverse=True)
    
    for item in line_items:
        print(f"\nSearching for: {item}")
        # 获取该项目的所有可能的字段名
        possible_fields = FIELD_MAPPINGS.get(item, [item])
        print(f"Possible field names: {possible_fields}")
        
        found = False
        for field in possible_fields:
            # 在现金流量表中查找
            if not cashflow.empty and field in cashflow.index:
                print(f"Found {field} in cash flow statement")
                for date in all_dates:
                    if date in cashflow.columns:
                        value = cashflow.loc[field, date]
                        if pd.notna(value):  # 只添加非空值
                            results.append({
                                "ticker": ticker,
                                "line_item": item,
                                "value": float(value),
                                "date": date.strftime("%Y-%m-%d"),
                                "period": period,
                                "source": "cashflow"
                            })
                found = True
                break
                
            # 在资产负债表中查找
            elif not balance.empty and field in balance.index:
                print(f"Found {field} in balance sheet")
                for date in all_dates:
                    if date in balance.columns:
                        value = balance.loc[field, date]
                        if pd.notna(value):  # 只添加非空值
                            results.append({
                                "ticker": ticker,
                                "line_item": item,
                                "value": float(value),
                                "date": date.strftime("%Y-%m-%d"),
                                "period": period,
                                "source": "balance"
                            })
                found = True
                break
                
            # 在利润表中查找
            elif not income.empty and field in income.index:
                print(f"Found {field} in income statement")
                for date in all_dates:
                    if date in income.columns:
                        value = income.loc[field, date]
                        if pd.notna(value):  # 只添加非空值
                            results.append({
                                "ticker": ticker,
                                "line_item": item,
                                "value": float(value),
                                "date": date.strftime("%Y-%m-%d"),
                                "period": period,
                                "source": "income"
                            })
                found = True
                break
                
        if not found:
            print(f"Not found in any statement")
    
    print(f"\nFound {len(results)} matching items")
    if results:
        print("Sample results:")
        for r in results[:3]:
            print(r)
    
    # 按日期排序（降序）
    results.sort(key=lambda x: x["date"], reverse=True)
    
    # 确保每个时期都有完整的数据
    dates = sorted(list(set(r["date"] for r in results)), reverse=True)
    complete_periods = []
    
    for date in dates:
        period_items = [r for r in results if r["date"] == date]
        if len(period_items) == len(line_items):  # 如果这个时期有所有需要的数据
            complete_periods.extend(period_items)
            if len(complete_periods) >= limit * len(line_items):  # 如果已经有足够的完整时期
                break
    
    return complete_periods

def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """获取内部交易数据
    
    Args:
        ticker: 股票代码
        end_date: 结束日期
        start_date: 开始日期（可选）
        limit: 返回的最大记录数
    
    Returns:
        list[InsiderTrade]: 内部交易记录列表
    """
    stock = yf.Ticker(ticker)
    # 获取内部购买数据
    purchases = stock.insider_purchases
    
    if purchases is None or purchases.empty:
        return []
    
    all_trades = []
    for _, trade in purchases.iterrows():
        # 转换日期为ISO格式字符串
        transaction_date = trade.get('Transaction Date', None)
        if transaction_date:
            transaction_date = pd.to_datetime(transaction_date).strftime('%Y-%m-%dT%H:%M:%S')
        
        filing_date = trade.get('Filing Date', None)
        if filing_date:
            filing_date = pd.to_datetime(filing_date).strftime('%Y-%m-%dT%H:%M:%S')
        
        insider_trade = InsiderTrade(
            ticker=ticker,
            issuer=stock.info.get('longName', None),
            name=trade.get('Insider Name', None),
            title=trade.get('Insider Title', None),  # 新API提供了职位信息
            is_board_director=None,  # 仍然无法确定是否为董事会成员
            transaction_date=transaction_date,
            transaction_shares=float(trade.get('Number of Shares', 0)) if not pd.isna(trade.get('Number of Shares')) else None,
            transaction_price_per_share=float(trade.get('Share Price', 0)) if not pd.isna(trade.get('Share Price')) else None,
            transaction_value=float(trade.get('Total Value', 0)) if not pd.isna(trade.get('Total Value')) else None,
            shares_owned_before_transaction=None,
            shares_owned_after_transaction=float(trade.get('Shares Owned', 0)) if not pd.isna(trade.get('Shares Owned')) else None,
            security_title=None,
            filing_date=filing_date or transaction_date
        )
        
        trade_date = transaction_date or filing_date
        if trade_date:
            if (start_date is None or trade_date >= start_date) and trade_date <= end_date:
                all_trades.append(insider_trade)
    
    # 按日期排序（降序）
    all_trades.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
    
    # 应用限制
    return all_trades[:limit]

def get_market_cap(ticker: str) -> float:
    """获取市值"""
    stock = yf.Ticker(ticker)
    return stock.info.get("marketCap", None)

def get_prices(
    ticker: str,
    start_date: str,
    end_date: str
) -> List[Dict[str, Any]]:
    """获取历史价格数据"""
    if not ticker:
        raise ValueError("Ticker cannot be None or empty")
    
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    
    prices = []
    for date, row in df.iterrows():
        prices.append({
            "time": date.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        })
    return prices

def prices_to_df(prices: List[Dict[str, Any]]) -> pd.DataFrame:
    """将价格数据转换为DataFrame"""
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df

def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """获取价格数据并转换为DataFrame"""
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)

def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """获取公司新闻"""
    stock = yf.Ticker(ticker)
    news_data = stock.news
    
    if not news_data:
        return []
    
    all_news = []
    for news in news_data:
        news_date = datetime.fromtimestamp(news.get('providerPublishTime', 0)).isoformat()
        
        news_item = CompanyNews(
            ticker=ticker,
            title=news.get('title', ''),
            author=news.get('author', ''),
            source=news.get('publisher', ''),
            date=news_date,
            url=news.get('link', ''),
            sentiment=None
        )
        
        if (start_date is None or news_date >= start_date) and news_date <= end_date:
            all_news.append(news_item)
    
    all_news.sort(key=lambda x: x.date, reverse=True)
    return all_news[:limit] 