import backtrader as bt
import pandas as pd
import numpy as np
import datetime
from copy import deepcopy
from datas import *
import warnings
from pandas.errors import SettingWithCopyWarning
import os
import matplotlib.pyplot as plt
from alphalens.utils import compute_forward_returns, get_clean_factor
import pandas as pd
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')  # 使用非交互式后端
matplotlib.rcParams['font.family'] = 'SimHei'

printlog = False

warnings.simplefilter(action='ignore', category=(FutureWarning, SettingWithCopyWarning))

# 数据准备
def prepare_data(year, start_date, end_date, alphaset, subset, alpha_name, list_assets):

    missing_assets = [
        "600005", "600068", "600432", "600832", 
        "601268", "601299", "601558", 
        "000024", "000527", "000562", "000780", 
        "600102", "600631", "600087", # 2011年添加上去的
    ]

    # 生成完整的交易日索引
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 频率为交易日

    list_assets = [asset for asset in list_assets if asset not in missing_assets]

    df_org= get_all_date_data(start_date, end_date, list_assets)

    df1 = df_org.rename(columns={
                            "date": "datetime", 
                            "asset": "sec_code"})

    df1["openinterest"] = 0

    daily_price=df1[['sec_code','datetime', "open", "close", "high", "low", "volume", 'openinterest']]

    daily_price['datetime'] = pd.to_datetime(daily_price['datetime'])

    # # 以 datetime 为 index，类型为 datetime 或 date 类型，Datafeeds 默认情况下是将 index 匹配给 datetime 字段；
    daily_price = daily_price.set_index(['datetime'])


    df_2 = df_org[['date', 'asset', "close"]]
    df_2['date'] = pd.to_datetime(df_2['date'])
    # print(df_all)

    close = df_2.pivot(index='date', columns='asset', values='close')
    

    # 读取已经计算好的因子
    alpha = pd.read_csv('alphas/{}/{}/{}.csv'.format(alphaset, subset, alpha_name))

    # 筛选出今年的数据，确保日期在start_date和end_date之间
    alpha = alpha[(alpha['date'] >= start_date) & (alpha['date'] <= end_date)] 

    # 获取close中的所有有效资产
    valid_assets = close.columns

    # 筛选出alpha中的有效资产
    alpha = alpha[['date'] + [col for col in alpha.columns if col in valid_assets]]

    # 因子矩阵转换为一维数据(alphalens需要的格式)
    alpha = alpha.melt(id_vars=['date'], var_name='asset', value_name='factor')

    # date列转为日期格式
    alpha['date'] = pd.to_datetime(alpha['date'])
    alpha = alpha[['date', 'asset', 'factor']]

    # 设置二级索引
    alpha = alpha.set_index(['date', 'asset'], drop=True)
    alpha.sort_index(inplace=True)
    
    
    ret1 = compute_forward_returns(alpha, close, periods=[1, 5, 10])

    ret = get_clean_factor(alpha, ret1, quantiles=5, max_loss=1)
    # ret = get_clean_factor_and_forward_returns(alpha, close, quantiles=5, max_loss=1, periods=[1, 5, 10])
    ret = ret.reset_index()
    ret = ret[ret['factor_quantile'] == 5]

    ret = ret[['date','asset']]
    ret['weight'] = 1/60
    trade_info = ret.rename(columns={""
            "date": "trade_date", 
            "asset": "sec_code"})

    return daily_price, trade_info, close

# 回测策略
class TestStrategy(bt.Strategy):
    params = (
        ('buy_stocks', None), # 传入各个调仓日的股票列表和相应的权重
    )
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
         
        self.trade_dates = pd.to_datetime(self.p.buy_stocks['trade_date'].unique()).tolist()
        
        # pd.DataFrame(self.trade_dates).to_csv('data/trade_dates.csv')
        
        self.buy_stock = self.p.buy_stocks # 保留调仓信息
        self.order_list = []  # 记录以往订单，在调仓日要全部取消未成交的订单
        self.buy_stocks_pre = [] # 记录上一期持仓
    
    def next(self):
        # 获取当前的回测时间点
        dt = self.datas[0].datetime.date(0)
        # 打印当前时刻的总资产
        if(printlog):
            self.log('当前总资产 %.2f' %(self.broker.getvalue()))
        # 如果是调仓日，则进行调仓操作
        
        dt = pd.Timestamp(dt)
        
        if dt in self.trade_dates:
            if(printlog):
                print(f"--------------{dt} 为调仓日----------")
            
            #取消之前所下的没成交也未到期的订单
            if len(self.order_list) > 0:
                if(printlog):
                    print("--------------- 撤销未完成的订单 -----------------")
                for od in self.order_list:
                    # 如果订单未完成，则撤销订单
                    self.cancel(od) 
                 #重置订单列表
                self.order_list = [] 

            # 提取当前调仓日的持仓列表
            buy_stocks_data = self.buy_stock.query(f"trade_date=='{dt}'")
            long_list = buy_stocks_data['sec_code'].tolist()
            if(printlog):
                print('long_list', long_list)  # 打印持仓列表

            # 对现有持仓中，调仓后不再继续持有的股票进行卖出平仓
            sell_stock = [i for i in self.buy_stocks_pre if i not in long_list]
            if(printlog):
                print('sell_stock', sell_stock)
            
            if sell_stock:
                if(printlog):
                    print("-----------对不再持有的股票进行平仓--------------")
                for stock in sell_stock:
                    data = self.getdatabyname(stock)
                    if self.getposition(data).size > 0 :
                        od = self.close(data=data)  
                        self.order_list.append(od) # 记录卖出订单

            # 买入此次调仓的股票：多退少补原则
            if(printlog):
                print("-----------买入此次调仓期的股票--------------")
            for stock in long_list:
                w = buy_stocks_data.query(f"sec_code=='{stock}'")['weight'].iloc[0] # 提取持仓权重
                data = self.getdatabyname(stock)
                order = self.order_target_percent(data=data, target=w*0.95) # 为减少可用资金不足的情况，留 5% 的现金做备用
                self.order_list.append(order)

            self.buy_stocks_pre = long_list  # 保存此次调仓的股票列表
        
    #订单日志    
    def notify_order(self, order):
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已被处理的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                if(printlog):
                    self.log(
                        'BUY EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s' %
                        (order.ref,
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm,
                        order.executed.size,
                        order.data._name))
            else:  # Sell
                if(printlog):
                    self.log('SELL EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s' %
                            (order.ref,
                            order.executed.price,
                            order.executed.value,
                            order.executed.comm,
                            order.executed.size,
                            order.data._name))

# 回测函数
def backtest_alpha(alpha_name, year, start_date, end_date, alphaset, subset,list_assets, strategy_name=TestStrategy):
    # 准备数据
    daily_price, trade_info, close = prepare_data(year, start_date, end_date, alphaset, subset, alpha_name, list_assets)
    
    # 实例化大脑
    cerebro_ = bt.Cerebro() 

    # 按股票代码，依次循环传入数据
    for stock in daily_price['sec_code'].unique():
        # 日期对齐
        data = pd.DataFrame(index=daily_price.index.unique())
        df = daily_price.query(f"sec_code=='{stock}'")[['open','high','low','close','volume','openinterest']]
        data_ = pd.merge(data, df, left_index=True, right_index=True, how='left')
        data_.loc[:,['volume','openinterest']] = data_.loc[:,['volume','openinterest']].fillna(0)
        data_.loc[:,['open','high','low','close']] = data_.loc[:,['open','high','low','close']].fillna(method='pad')
        
        datafeed = bt.feeds.PandasData(dataname=data_, fromdate=pd.Timestamp(start_date), todate=pd.Timestamp(end_date))
        cerebro_.adddata(datafeed, name=stock)
        # print(f"{stock} Done !") 
    
    cerebro = deepcopy(cerebro_)  # 深度复制已经导入数据的 cerebro_，避免重复导入数据 
    # 初始资金 100,000,000    
    cerebro.broker.setcash(100000.0) 
    # cerebro.broker.setcommission(commission=0.0015)
    # 添加策略
    cerebro.addstrategy(strategy=strategy_name, buy_stocks=trade_info) # 通过修改参数 buy_stocks ，使用同一策略回测不同的持仓列表

    # 添加分析指标
    # 返回年初至年末的年度收益率
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')
    # 计算最大回撤相关指标
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')
    # 计算年化收益
    cerebro.addanalyzer(bt.analyzers.Returns, _name='_Returns', tann=252)
    # 计算年化夏普比率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio', timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0) # 计算夏普比率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='_SharpeRatio_A')
    # 返回收益率时序
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')
    
    # 执行回测
    results = cerebro.run()
    strat = results[0]

    # 假设 strat.analyzers._TimeReturn.get_analysis() 已经返回了一个pandas Series对象
    ret = pd.Series(strat.analyzers._TimeReturn.get_analysis())
    # ret.to_csv(f'results/{alpha_name}_TimeReturn.csv')
    
    ret1 = [alpha_name,  # 因子名称
           year,  # 年度'
           strategy_name.__name__,  # 策略名称
           strat.analyzers._Returns.get_analysis()['rtot'],  # 收益率
           strat.analyzers._Returns.get_analysis()['ravg'],  # 日均收益率
           strat.analyzers._Returns.get_analysis()['rnorm'],  # 年化收益率
           strat.analyzers._DrawDown.get_analysis()['max']['drawdown'] * (-1), # 最大回撤 
           strat.analyzers._SharpeRatio.get_analysis()['sharperatio']] # 夏普比率
    
    return ret, ret1

def get_alpha_list(directory):
    """
    从指定文件夹中获取所有文件名，去掉后缀名，并返回一个文件名列表。

    参数:
        directory (str): 文件夹路径

    返回:
        list: 去掉后缀名的文件名列表
    """
    try:
        # 列出文件夹中的所有文件和子文件夹
        files = os.listdir(directory)
        # 过滤掉子文件夹，只保留文件，并去掉后缀
        alphalist = [os.path.splitext(f)[0] for f in files if os.path.isfile(os.path.join(directory, f))]
        return alphalist
    except FileNotFoundError:
        print(f"错误：文件夹 {directory} 不存在！")
        return []
    except Exception as e:
        print(f"发生错误: {e}")
        return []

def plot_alpha_results(alpha_name, output_dir, plot_results):
    """
    绘制因子回测结果的图像，并保存到指定文件夹中。

    参数:
        alpha_name (str): 因子名称
        output_dir (str): 图像保存路径
        plot_results (dict): 回测结果字典
    """
    # colors = ['skyblue', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    
    plt.figure(figsize=(12, 6))
    plt.title(f"{alpha_name} 的累积收益率")
    
    for strategy, ret in plot_results.items():
        
        # 计算累计收益率并绘图
        plot_result = (ret + 1).cumprod()
        plot_result.plot(label=strategy.__name__) #, color=colors.pop(0)

    # 保存图像
    output_path = f'{output_dir}/{alpha_name}.png'
    plt.savefig(output_path)
    print(f"图像已保存到: {output_path}")

    # 清理图像，避免内存问题
    plt.clf()


# 主程序
if __name__ == "__main__":
    year = 2018
    start_date = '2018-04-30'
    end_date = '2023-04-30'
    
    # alphaset = 'ourAlphas'
    subset = '20110430'
    
    alphaset = 'multialpha'
    
    alpha_names = get_alpha_list(f'alphas/{alphaset}/{subset}')
    alpha_names = ['alpha_ch3', 'alpha_famafrench']
    
    strategy_list = [TestStrategy]
    
    list_assets, df_assets = get_hs300_stocks(f'{year}-01-01')

    # 设置保存文件夹
    output_dir = "output_charts/multi/"
    result_dir = "results"
    
    # 如果文件夹不存在，则创建文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    results = []

    for alpha_name in tqdm(alpha_names):
        
        plot_results = {}
        
        for strategy in strategy_list:
            print(f"开始回测 {alpha_name}...的策略： {strategy.__name__}")
            ret, ret1 = backtest_alpha(alpha_name, year, start_date, end_date, alphaset, subset,list_assets, strategy)
            
            results.append(ret1) # 将回测结果添加到结果列表中
            plot_results[strategy] = ret # 将回测结果添加到绘图结果字典中
            
        plot_alpha_results(alpha_name, output_dir, plot_results)
        print(f"回测 {alpha_name} 完成！")    
        
    
    results = pd.DataFrame(results, columns = ['alpha','年度', '策略名称','收益率', '日均收益率', '年化收益率', '最大回撤(%)', '夏普比率'])
    
    results.to_csv(f'{result_dir}/results_2012_2023_multi.csv', index=False)
    
    
    
    



