import baostock as bs
import pandas as pd
from multiprocessing import Pool
import akshare as ak
import os

def download_date_data(code, flag):
    try:
        fg = '' if flag not in ['qfq', 'hfq'] else flag
        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date='20090101',end_date='20231231', adjust=fg)
        stock_zh_a_hist_df.to_csv(f'./data_{flag}/{code}.csv')
    except Exception as e:
        print(f"download {flag} stock {code} error!!!")

def download_all_date_data(flag):
    # 获取所有股票代码，akshare接口
    stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()
    list_code = stock_zh_a_spot_em_df['代码'].to_list()

    fg = 'bfq' if flag not in ['qfq', 'hfq'] else flag # bfq: 不复权; qfq: 前复权; hfq: 后复权
    # 创建保存路径
    path = f'data_{fg}'
    if not os.path.isdir(path):
        os.makedirs(path)

    # 创建进程池来下载股票日数据
    count = os.cpu_count()
    pool = Pool(min(count*4, 60))
    for code in list_code:
        pool.apply_async(download_date_data, (code, flag))

    pool.close()
    pool.join()

def get_all_date_data(start_time, end_time, list_assets):
    data_path = 'data_bfq'

    # 从本地保存的数据中读出需要的股票日数据
    list_all = []
    for c in list_assets:
        df = pd.read_csv(f'{data_path}/{c}.csv')
        df['asset'] = c
        list_all.append(df[(df['日期'] >= start_time) & (df['日期'] <= end_time)])
        
    print(len(list_all))

    # 所有股票日数据拼接成一张表
    df_all = pd.concat(list_all)
        
    # 修改列名
    df_all = df_all.rename(columns={
        "日期": "date", 
        "开盘": "open", 
        "收盘": "close", 
        "最高": "high", 
        "最低": "low", 
        "成交量": "volume", 
        "成交额": "amount",
        "涨跌幅": "pctChg"})
    # 计算平均成交价
    df_all['vwap'] =  df_all.amount / df_all.volume / 100

    # 返回计算因子需要的列
    df_all = df_all.reset_index()
    df_all = df_all[['asset','date', "open", "close", "high", "low", "volume", 'vwap', "pctChg"]]
    return df_all

def get_zz500_stocks(time):
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    # 获取中证500成分股
    rs = bs.query_zz500_stocks('2019-01-01')
    print('query_zz500 error_code:'+rs.error_code)
    print('query_zz500  error_msg:'+rs.error_msg)

    # 打印结果集
    zz500_stocks = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        zz500_stocks.append(rs.get_row_data())
    result = pd.DataFrame(zz500_stocks, columns=rs.fields)

    lists = result['code'].to_list()
    lists = [x.split('.')[1] for x in lists]

    # 登出系统
    bs.logout()
    return lists, result

# 获取沪深300成分股
def get_hs300_stocks(time):
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    # 获取沪深300成分股
    rs = bs.query_hs300_stocks(time)
    print('query_hs300 error_code:'+rs.error_code)
    print('query_hs300  error_msg:'+rs.error_msg)

    # 打印结果集
    hs300_stocks = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        hs300_stocks.append(rs.get_row_data())
    result = pd.DataFrame(hs300_stocks, columns=rs.fields)
    
    lists = result['code'].to_list()
    lists = [x.split('.')[1] for x in lists]

    # 登出系统
    bs.logout()
    return lists, result

# 下载指数数据
def download_index_data(code):
    path = 'index'
    stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol=code)
    # 创建保存路径
    if not os.path.isdir(path):
        os.makedirs(path)
    stock_zh_index_daily_df.to_csv(f'{path}/{code}.csv')

# 生成季度日期列表
def generate_quarterly_dates(start_date, end_date):
    """
    生成从 start_date 到 end_date 的季度日期列表。
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='Q')
    return [date.strftime('%Y%m%d') for date in dates]

# 下载利润表数据
def download_profit_sheet_data(start_date, end_date):
    """
    下载指定时间范围内的利润表数据，并分别保存为 CSV 文件。
    文件名为日期，存储在指定文件夹中。
    """
    path = 'profit_sheet'
    if not os.path.isdir(path):
        os.makedirs(path)

    # 获取季度日期列表
    date_list = generate_quarterly_dates(start_date, end_date)

    for date in date_list:
        try:
            # 下载数据
            data = ak.stock_lrb_em(date=date)
            data.to_csv(f'{path}/{date}.csv')
        except Exception as e:
            print(f"下载 {date} 的利润表数据失败！")
            print(e)
            
# 下载资产负债表数据
def download_balance_sheet_data(start_date, end_date):
    """
    下载指定时间范围内的资产负债表数据，并分别保存为 CSV 文件。
    文件名为日期，存储在指定文件夹中。
    """
    path = 'balance_sheet'
    if not os.path.isdir(path):
        os.makedirs(path)

    # 获取季度日期列表
    date_list = generate_quarterly_dates(start_date, end_date)

    for date in date_list:
        try:
            # 下载数据
            data = ak.stock_zcfz_em(date=date)
            data2 = ak.stock_zcfz_bj_em(date=date)
            data = pd.concat([data, data2])
            data.to_csv(f'{path}/{date}.csv')
        except Exception as e:
            print(f"下载 {date} 的资产负债表数据失败！")
            print(e) 

# 下载现金流量表数据            
def download_cash_flow_data(start_date, end_date):
    """
    下载指定时间范围内的现金流量表数据，并分别保存为 CSV 文件。
    文件名为日期，存储在指定文件夹中。
    """
    path = 'cash_flow'
    if not os.path.isdir(path):
        os.makedirs(path)

    # 获取季度日期列表
    date_list = generate_quarterly_dates(start_date, end_date)

    for date in date_list:
        try:
            # 下载数据
            data = ak.stock_xjll_em(date=date)
            data.to_csv(f'{path}/{date}.csv')
            
        except Exception as e:
            print(f"下载 {date} 的现金流量表数据失败！")
            print(e)

# 下载无风险利率数据
def download_risk_free_rate(start_date, end_date):
    """
    下载指定时间范围内的无风险利率数据，并保存为 CSV 文件。
    """
    path = 'index'
    if not os.path.isdir(path):
        os.makedirs(path)

    try:
        # 隔夜拆借利率
        data = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator='隔夜')
        data.rename(columns={'报告日': '日期', '利率':'无风险利率'}, inplace=True)
        data.to_csv(f'{path}/Shibor_Overnight.csv')
        
    except Exception as e:
        print(f"下载无风险利率数据失败！")
        print(e)

if __name__ == '__main__':
    start_date = '20090101'
    end_date = '20231231'
    download_index_data("sh000300")
    download_all_date_data("bfq")
    download_profit_sheet_data(start_date, end_date)
    download_balance_sheet_data(start_date, end_date)
    download_cash_flow_data(start_date, end_date)
    download_risk_free_rate(start_date, end_date)
    
