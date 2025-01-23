"""
趋势分析可视化测试模块

该模块提供了直接运行的测试用例，用于测试和展示技术分析可视化功能。
可以通过命令行参数配置不同的测试场景。

使用方法：
在项目根目录下运行：
python -m src.analysis.test_visualize [参数]

示例：
python -m src.analysis.test_visualize --symbol AAPL --start 2023-01-01 --end 2024-01-01 --window 200
"""

import argparse
from datetime import datetime, timedelta

from tools.yfinance_api import get_prices, prices_to_df
from analysis.visualize import visualize_trend_signals

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='技术分析可视化测试工具')
    
    # 添加命令行参数
    parser.add_argument('--symbol', type=str, default='AAPL',
                      help='股票代码 (默认: AAPL)')
    
    parser.add_argument('--start', type=str, 
                      default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                      help='开始日期 (默认: 一年前)')
    
    parser.add_argument('--end', type=str,
                      default=datetime.now().strftime('%Y-%m-%d'),
                      help='结束日期 (默认: 今天)')
    
    parser.add_argument('--window', type=int, default=200,
                      help='显示窗口大小 (默认: 200天)')
    
    # 添加信号显示控制参数
    parser.add_argument('--trend', action='store_true', default=False,
                      help='显示趋势信号')
    parser.add_argument('--mean-reversion', action='store_true', default=False,
                      help='显示均值回归信号')
    parser.add_argument('--momentum', action='store_true', default=True,
                      help='显示动量信号')
    parser.add_argument('--volatility', action='store_true', default=False,
                      help='显示波动率信号')
    parser.add_argument('--all-signals', action='store_true', default=False,
                      help='显示所有信号')
    parser.add_argument('--combined', action='store_true', default=False,
                      help='显示综合信号')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    print(f"\n=== 技术分析可视化测试 ===")
    print(f"股票代码: {args.symbol}")
    print(f"时间范围: {args.start} 至 {args.end}")
    print(f"显示窗口: {args.window}天")
    print("信号显示设置:")
    print(f"- 趋势信号: {'是' if args.trend or args.all_signals else '否'}")
    print(f"- 均值回归信号: {'是' if args.mean_reversion or args.all_signals else '否'}")
    print(f"- 动量信号: {'是' if args.momentum or args.all_signals else '否'}")
    print(f"- 波动率信号: {'是' if args.volatility or args.all_signals else '否'}")
    print(f"- 综合信号: {'是' if args.combined or args.all_signals else '否'}")
    print("========================\n")
    
    try:
        # 获取股票数据
        prices = get_prices(
            ticker=args.symbol,
            start_date=args.start,
            end_date=args.end
        )
        
        if not prices:
            print(f"错误: 无法获取 {args.symbol} 的数据")
            return
        
        # 转换为DataFrame
        prices_df = prices_to_df(prices)
        
        print(f"获取到 {len(prices_df)} 个交易日的数据")
        print(f"数据范围: {prices_df.index.min()} 至 {prices_df.index.max()}\n")
        
        # 显示可视化分析
        visualize_trend_signals(
            prices_df, 
            window_size=args.window,
            show_trend=args.trend or args.all_signals,
            show_mean_reversion=args.mean_reversion or args.all_signals,
            show_momentum=args.momentum or args.all_signals,
            show_volatility=args.volatility or args.all_signals,
            show_combined=args.combined or args.all_signals
        )
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return

if __name__ == "__main__":
    main() 