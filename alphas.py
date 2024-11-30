import pandas as pd
from multiprocessing import Pool
from datas import *
import os
import traceback
import time
from datetime import datetime

class Alphas(object):
    def __init__(self, df_data):
        pass

    @classmethod
    def calc_alpha(cls, path, func, data):
        try:
            t1 = time.time()
            res = func(data)
            res.to_csv(path)
            t2 = time.time()
            print(f"Factory {os.path.splitext(os.path.basename(path))[0]} time {t2-t1}")
        except Exception as e:
            print(f"generate {path} error!!! Error: {str(e)}")
            # traceback.print_exc()

    @classmethod
    def get_stocks_data(cls, start_date, end_date, list_assets, benchmark):
        # 截取向前一年和向后一年的数据，并改写成时间格式的字符串
        start_time = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
        end_time = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')

        index_path = 'index'
        df = pd.read_csv(f'{index_path}/{benchmark}.csv')
        bm_data =  df[(df['date'] >= start_time) & (df['date'] <= end_time)]

        # 修改列名
        bm_data = bm_data.rename(columns={
            "date": "benchmark_date", 
            "open": "benchmark_open", 
            "close": "benchmark_close", 
            "high": "benchmark_high", 
            "low": "benchmark_low", 
            "volume": "benchmark_vol"})

        data_path = 'data_merged'

        # 从本地保存的数据中读出需要的股票日数据
        list_all = []
        for c in list_assets:
            
            # 如果股票数据不存在，跳过
            if not os.path.exists(f'{data_path}/{c}.csv'):
                print(f'{data_path}/{c}.csv not exists')
                continue
            
            df = pd.read_csv(f'{data_path}/{c}.csv')
            df['asset'] = c # 设asset为股票代码
            df = df[(df['日期'] >= start_time) & (df['日期'] <= end_time)]
            df = df.merge(bm_data, how='outer', left_on='日期', right_on='benchmark_date')
            list_all.append(df)
            
        print(len(list_all))

        # 所有股票日数据拼接成一张表
        df_all = pd.concat(list_all)
        
        df_all['roe'] = df_all['净利润'] / df_all['股东权益合计']
        df_all['roa'] = df_all['净利润'] / df_all['资产-总资产']
        df_all['cvd'] = (
            df_all['成交量'].rolling(window=120).std() / 
            df_all['成交量'].rolling(window=120).mean()
        )
        # 换手率百分比转为小数
        df_all['换手率_小数'] = df_all['换手率'] / 100
        # 估算市值
        df_all['市值'] = df_all['成交额'] / df_all['换手率_小数']
        df_all['epq'] = (df_all['营业利润'] - df_all['利润总额'] * 0.25) / df_all['市值']
        df_all['企业价值'] = (
            df_all['股东权益合计'] + 
            df_all['负债-总负债'] - 
            df_all['资产-货币资金']
        )
        df_all['emq'] = df_all['企业价值'] / df_all['营业利润']
        df_all['sgq'] = (
            df_all['营业总收入'].rolling(window=4).mean() / 
            df_all['营业总收入'].shift(4) - 1
        )
        df_all['资产流动性'] = (
            df_all['资产-货币资金'] + 
            0.75 * df_all['资产-应收账款'] + 
            0.5 * df_all['资产-存货']
        )
        df_all['alaq'] = df_all['资产流动性'] / df_all['资产-总资产'].shift(1)
        df_all['pmq'] = df_all['营业利润'] / df_all['营业总收入'].shift(1)
        df_all['cta'] = df_all['资产-货币资金'] / df_all['资产-总资产']
        df_all['bm'] = df_all['股东权益合计'] / df_all['市值']
        df_all['ep'] = df_all['净利润'] / df_all['市值']
        
        # 修改列名
        df_all = df_all.rename(columns={
            "日期": "date", 
            "开盘": "open", 
            "收盘": "close", 
            "最高": "high", 
            "最低": "low", 
            "成交量": "volume", 
            "成交额": "amount",
            "涨跌幅": "pctChg",
            "换手率": "turnover", 
            "市值": "size",
            "市场溢酬因子__流通市值加权_Rmrf_tmv":"Rmrf",
            "市值因子__流通市值加权_Smb_tmv":"Smb",
            "账面市值比因子__流通市值加权_Hml_tmv":"Hml",
            "无风险利率": "rf"
            })
        df_all['turnover']  = df_all['turnover']/100
        df_all['vwap'] =  df_all.amount / df_all.volume / 100 # 计算平均成交价

        # 返回计算因子需要的列
        df_all = df_all.reset_index()
        df_all = df_all[['asset', 'date', "open", "close", "high", "low", "volume", "amount", 'vwap', "pctChg", 'turnover', 'benchmark_open', 'benchmark_close', 'benchmark_high', 'benchmark_low', 'benchmark_vol', 'roe', 'roa', 'cvd', 'epq', 'emq', 'sgq', 'alaq', 'pmq', 'cta', 'size', 'Rmrf', 'Smb', 'Hml', 'rf', 'bm', 'ep']]
        # ddu = df_all[df_all.duplicated()]
        df_all=df_all[df_all['asset'].notnull()]
        
        df_all.to_csv('data/df_all.csv')
        
        df_all = df_all.pivot(index='date', columns='asset')
        
        df_all.to_csv('data/df_all_pivot.csv')
        
        return df_all 
    
    @classmethod
    def get_benchmark(cls, start_date, end_date, code):
        # yer = int(year)
        # start_time = f'{yer-1}-01-01'
        # end_time = f'{yer+1}-01-01'
        start_time = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
        end_time = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')

        data_path = 'index'
        df = pd.read_csv(f'{data_path}/{code}.csv')
        return df[(df['date'] >= start_time) & (df['date'] <= end_time)]

    @classmethod
    def get_alpha_methods(cls, self):
        return (list(filter(lambda m: m.startswith("alpha") and callable(getattr(self, m)),
                            dir(self))))
    
    @classmethod
    def generate_alpha_single(cls, alpha_name, start_date, end_date, list_assets, benchmark, need_save=False):
        # 获取计算因子所需股票数据
        stock_data = cls.get_stocks_data(start_date, end_date, list_assets, benchmark)

        # 实例化因子计算的对象
        stock = cls(stock_data)

        factor = getattr(cls, alpha_name)
        if factor is None:
            print('alpha name is error!!!')
            return None
        
        alpha_data = factor(stock)

        if need_save:
            year = start_date.split('-')[0]
            path = f'alphas/{cls.__name__}/{year}'
            if not os.path.isdir(path):
                os.makedirs(path)
            alpha_data.to_csv(f'{path}/{alpha_name}.csv')

        return alpha_data
            

    @classmethod
    def generate_alphas(cls, start_date, end_date, list_assets, benchmark):
        t1 = time.time()
        # 获取计算因子所需股票数据
        stock_data = cls.get_stocks_data(start_date, end_date, list_assets, benchmark)

        # 实例化因子计算的对象
        stock = cls(stock_data)
        
        year = start_date.split('-')[0]
        
        # 因子计算结果的保存路径
        path = f'alphas/{cls.__name__}/{year}'

        # 创建保存路径
        if not os.path.isdir(path):
            os.makedirs(path)

        # 创建线程池
        count = os.cpu_count()
        pool = Pool(count)

        # 获取所有因子计算的方法
        methods = cls.get_alpha_methods(cls)
        
        # print(f"Starting to generate alphas for {methods}...")

        # 在线程池中计算所有alpha
        for m in methods:
            factor = getattr(cls, m)
            print(f"Starting to generate alpha for {m}...")
            try:
                pool.apply_async(cls.calc_alpha, (f'{path}/{m}.csv', factor, stock))
                print(f"generate {m} success!!!")
            except Exception as e:
                print(f"Error while calculating alpha for {m}: {str(e)}")
                traceback.print_exc()

        pool.close()
        pool.join()
        t2 = time.time()
        print(f"Total time {t2-t1}")