import numpy as np
import pandas as pd
from alphas import Alphas
from datas import *
from utils import *

# 'asset', 'date', "open", "close", "high", "low", "volume", "amount", 'vwap', "pctChg", 'turnover', 'benchmark_open', 'benchmark_close', 'benchmark_high', 'benchmark_low', 'benchmark_vol', 'roe', 'roa', 'cvd', 'epq', 'emq', 'sgq', 'alaq', 'pmq', 'cta', 'size'

class ourAlphas(Alphas):
    def __init__(self, df_data):
        self.open = df_data['open'] # 开盘价
        self.high = df_data['high'] # 最高价
        self.low = df_data['low'] # 最低价
        self.close = df_data['close'] # 收盘价
        self.volume = df_data['volume'] # 成交量
        self.returns = returns(df_data['close']) # 日收益
        self.vwap = df_data['vwap']  # 成交均价
        self.roe = df_data['roe'] # 股本回报率
        self.roa = df_data['roa'] # 资产回报率
        self.cvd = df_data['cvd'] # 成交量波动率
        self.epq = df_data['epq'] # 每市值盈利
        self.emq = df_data['emq'] # 企业价值倍数
        self.sgq = df_data['sgq'] # 季度销售增长率
        self.alaq = df_data['alaq'] # 流动资产占比
        self.pmq = df_data['pmq'] # 季度利润率
        self.cta = df_data['cta'] # 现金占比
        self.market_return = returns(df_data['benchmark_close']) # 市场收益率
        self.size = df_data['size'] # 市值
        self.turnover = df_data['turnover'] # 换手率
        self.amount = df_data['amount'] # 成交额
        self.pctChg = df_data['pctChg'] # 涨跌幅
    
    # 因子1：基于 ROE 的波动性因子
    def alpha_ROE(self):
        """
        alpha_ROE: 构造基于 ROE 的一个动态因子
        逻辑：
        1. 计算 ROE 的滚动均值和标准差，反映其稳定性。
        2. 用当前 ROE 与均值的偏离程度（标准化形式）作为因子值。
        3. 最后对因子值进行排名处理。
        """
        # 滚动计算 ROE 的均值和标准差（20 天窗口）
        mean_roe = mean(self.roe, 20)  # 滚动均值
        std_roe = stddev(self.roe, 20)  # 滚动标准差

        # 计算当前 ROE 相对均值的偏离程度（标准化值）
        standardized_roe = (self.roe - mean_roe) / (std_roe + 1e-5)

        # 排名并返回因子值
        return rank(standardized_roe)
    
    # 因子2：基于 ROA 的短期动量
    def alpha_ROA_momentum(self):
        """
        alpha_ROA_momentum: 反映 ROA 的短期变化趋势
        逻辑：
        1. 计算 ROA 的短期变化（差分），捕捉盈利能力变化的方向。
        2. 结合当前 ROA 排名，强化动量的相对位置效应。
        3. 返回动量值的排名作为因子值。
        """
        # 计算 ROA 的 5 天变化
        roa_delta = delta(self.roa, 5)

        # 结合 ROA 当前排名计算动量值
        momentum = roa_delta * rank(self.roa)

        # 返回因子值
        return rank(momentum)
    
    # 因子3：成交量与收盘价相关性
    def alpha_Volume_Close(self):
        """
        alpha_Volume_Close: 捕捉成交量与价格的动态关系
        逻辑：
        1. 计算成交量与收盘价的滚动相关性（10 天窗口）。
        2. 高相关性表明成交量驱动价格波动，反映市场活跃度。
        3. 对相关性进行排名处理，返回因子值。
        """
        # 计算成交量与收盘价的滚动相关性
        corr_volume_close = corr(rank(self.volume), rank(self.close), 10)

        # 返回因子值
        return rank(corr_volume_close)
    
    # 因子4：成交均价与收盘价差异
    def alpha_Price_VWAP(self):
        """
        alpha_Price_VWAP: 衡量价格强弱的因子
        逻辑：
        1. 计算当前收盘价与成交均价的差异（相对比例）。
        2. 收盘价高于均价表明市场需求强，反之需求弱。
        3. 返回比例差的排名作为因子值。
        """
        # 计算收盘价与成交均价的差异
        price_strength = (self.close - self.vwap) / (self.vwap + 1e-5)

        # 返回因子值
        return rank(price_strength)

    # 因子5：高价和低价的动态范围
    def alpha_High_Low(self):
        """
        alpha_High_Low: 捕捉市场波动的因子
        逻辑：
        1. 计算高价和低价的差值，反映单日波动范围。
        2. 取差值的滚动均值（20 天窗口），衡量短期波动趋势。
        3. 返回均值的排名作为因子值。
        """
        # 计算高价和低价的差值
        high_low_spread = self.high - self.low

        # 滚动计算差值均值
        spread_mean = mean(high_low_spread, 20)

        # 返回因子值
        return rank(spread_mean)

    # 因子6：基于季度销售增长率的波动性
    def alpha_SGQ_volatility(self):
        """
        alpha_SGQ_volatility: 衡量季度销售增长的稳定性
        逻辑：
        1. 计算季度销售增长率的滚动标准差（20 天窗口）。
        2. 稳定的增长率通常与低风险和持续盈利能力相关。
        3. 返回波动率的排名作为因子值。
        """
        # 滚动计算季度销售增长率的标准差
        sgq_stddev = stddev(self.sgq, 20)

        # 返回因子值
        return rank(sgq_stddev)

    # 因子7：基于季度利润率的动量
    def alpha_PMQ_momentum(self):
        """
        alpha_PMQ_momentum: 捕捉季度利润率的变化趋势
        逻辑：
        1. 计算季度利润率的短期变化（10 天差分），反映利润率的动态。
        2. 结合季度利润率的排名，提升因子稳定性。
        3. 返回动量值的排名作为因子值。
        """
        # 计算季度利润率的短期变化
        pmq_delta = delta(self.pmq, 10)

        # 动量计算
        momentum = pmq_delta * rank(self.pmq)

        # 返回因子值
        return rank(momentum)

    # 因子8：ROE 和 ROA 的综合盈利因子
    def alpha_ROE_ROA(self):
        """
        alpha_ROE_ROA: 综合反映公司盈利能力的因子
        逻辑：
        1. 分别计算 ROE 和 ROA 的排名值。
        2. 将两者相乘，形成综合盈利能力信号。
        3. 返回综合信号的排名作为因子值。
        """
        # 计算 ROE 和 ROA 的排名
        roe_rank = rank(self.roe)
        roa_rank = rank(self.roa)

        # 综合盈利信号
        combined_profitability = roe_rank * roa_rank

        # 返回因子值
        return rank(combined_profitability)

    # 因子9：市值因子
    def alpha_size(self):
        """
        alpha_size: 市值因子，单位为千元人民币
        逻辑：
        1. 估算流通市值：成交均价（VWAP）乘以成交量。
        2. 使用对数化处理市值，减少极端值影响。
        3. 对对数市值进行排名，生成因子值。
        公式：
        市值 = max(成交均价 * 成交量, 1e-5)  # 防止为零的情况
        log_size = log(市值)  # 对市值取对数
        alpha_size = rank(log_size)  # 对数市值排名
        """
        market_cap = np.maximum(self.vwap * self.volume / 100000, 1e-5)  # 估算流通市值（单位：千元）
        log_size = np.log(market_cap)  # 对市值取对数
        return rank(log_size)  # 返回排名

    # 因子10： 12-2 月动量因子 (Momentum Factor)
    def alpha_momentum(self):
        """
        alpha_momentum: 动量因子
        逻辑：
        1. 计算过去 12 个月的累计收益（252 个交易日）。
        2. 减去过去 1 个月的累计收益（21 个交易日），去除短期影响。
        3. 对计算结果进行排名，生成因子值。
        公式：
        r12m = 过去 12 个月累计收益 = ts_sum(returns, 252)
        r1m = 过去 1 个月累计收益 = ts_sum(returns, 21)
        alpha_momentum = rank(r12m - r1m)  # 12个月动量减去1个月动量后排名
        """
        cumulative_return_12m = ts_sum(self.returns, 252)  # 计算过去 12 个月的累计收益
        cumulative_return_1m = ts_sum(self.returns, 21)  # 计算过去 1 个月的累计收益
        return rank(cumulative_return_12m - cumulative_return_1m)  # 计算并排名

    # 因子11：基于 CAPM 模型计算的 β 系数 (Beta Factor)
    def alpha_beta(self):
        """
        alpha_beta: CAPM β 系数
        逻辑：
        1. 对股票收益率与市场收益率进行线性回归。
        2. 提取回归斜率 β 作为因子值。
        3. 对 β 进行排名，生成因子值。
        公式：
        beta = regression_slope(returns, market_returns)  # 计算线性回归的斜率β
        alpha_beta = rank(beta)  # 对 β 系数进行排名
        """
        beta = regression_slope(self.returns, self.market_return)  # 计算 CAPM β 系数
        return rank(beta)  # 对 β 进行排名

    # 因子12：月度换手率因子 (Turnover Factor)
    def alpha_turnover(self):
        """
        alpha_turnover: 月度换手率因子
        逻辑：
        1. 对换手率进行排名，生成因子值。
        公式：
        alpha_turnover = rank(turnover)  # 对月度换手率进行排名
        """
        return rank(self.turnover)  # 直接对月度换手率进行排名

    # 因子13：短期反转因子 (Short-Term Reversal Factor)
    def alpha_reversal(self):
        """
        alpha_reversal: 短期反转因子
        逻辑：
        1. 计算过去 1 个月（21 个交易日）的累计收益。
        2. 对累计收益取负值，捕捉短期反转效应。
        3. 对反转因子值进行排名，生成最终因子值。
        公式：
        srev_1m = -ts_sum(returns, 21)  # 过去1个月累计收益的负值
        alpha_reversal = rank(srev_1m)  # 对反转因子进行排名
        """
        monthly_return = ts_sum(self.returns, 21)  # 计算过去 1 个月的累计收益
        return rank(-monthly_return)  # 取负值并排名

    # 因子14：流动性因子 (Liquidity Factor)
    def alpha_liquidity(self):
        """
        alpha_liquidity: 流动性因子
        逻辑：
        1. 计算资产流动性指标，反映公司流动性水平。
        2. 对流动性指标进行排名，生成因子值。
        公式：
        alpha_liquidity = rank(资产流动性)  # 对资产流动性指标进行排名
        """
        return rank(self.alaq)  # 直接对资产流动性指标进行排名
    
    # 因子15：基于 CTA 模型的现金占比因子 (Cash-to-Assets Factor)
    def alpha_CTA(self):
        """
        alpha_CTA: 现金占比因子
        逻辑：
        1. 计算现金占比（CTA），反映公司现金流动性。
        2. 对现金占比进行排名，生成因子值。
        公式：
        alpha_CTA = rank(CTA)  # 对现金占比进行排名
        """
        return rank(self.cta)
    
        # Alpha#1	 (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)
    
    # 因子16：alpha001
    def alpha001(self):
        inner = self.close.copy()
        inner[self.returns < 0] = stddev(self.returns, 20)
        return rank(ts_argmax(inner ** 2, 5)) - 0.5
    
    
        
if __name__ == '__main__':
    year = '2013'
    start_date = '20130430'
    end_date = '20220430'
    
    list_assets,df_asserts = get_hs300_stocks(f"{year}-01-01")

    ################ 计算所有 #################   
    ourAlphas.generate_alphas(start_date, end_date, list_assets,"sh000300")
    
    # ret = ourAlphas.generate_alpha_single('alpha001', year, list_assets, "sh000300", True)
    # print(ret)