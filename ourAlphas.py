import numpy as np
import pandas as pd
from alphas import Alphas
from datas import *
from utils import *

# 'asset', 'date', "open", "close", "high", "low", "volume", "amount", 'vwap', "pctChg", 'turnover', 'benchmark_open', 'benchmark_close', 'benchmark_high', 'benchmark_low', 'benchmark_vol', 'roe', 'roa', 'cvd', 'epq', 'emq', 'sgq', 'alaq', 'pmq', 'cta', 'size'

def calculate_returns(self, condition, weighted):
    """
    计算每个时间点对应的股票组合收益率
    :param condition: 满足条件的布尔矩阵
    :param weighted: 是否使用市值加权平均
    """
    def calc(row, condition_row):
        selected_stocks = row.index[condition_row]  # 获取符合条件的股票
        if not selected_stocks.any():  # 如果没有满足条件的股票
            return None
        if weighted:
            weights = self.size.loc[row.name, selected_stocks]
            weights /= weights.sum()  # 归一化权重
            return (row[selected_stocks] * weights).sum()
        else:
            return row[selected_stocks].mean()

    return self.returns.apply(lambda row: calc(row, condition.loc[row.name]), axis=1)

# 是否为市值加权平均
weight = False

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
        self.bm = df_data['bm'] # 账面市值比
        self.rf = df_data['rf'] # 无风险利率
        self.ep = df_data['ep'] # 每股收益

        self.Rmrf = df_data['Rmrf'] # 市场溢酬因子
        self.Smb = df_data['Smb'] # 市值因子
        self.Hml = df_data['Hml'] # 账面市值比因子

        self.returns.to_csv('data/returns.csv')
        self.market_return.to_csv('data/market_return.csv')
    
    # # # 因子1：基于 ROE 的波动性因子
    # # def alpha_ROE(self):
    #     """
    #     alpha_ROE: 构造基于 ROE 的一个动态因子
    #     逻辑：
    #     1. 计算 ROE 的滚动均值和标准差，反映其稳定性。
    #     2. 用当前 ROE 与均值的偏离程度（标准化形式）作为因子值。
    #     3. 最后对因子值进行排名处理。
    #     """
    #     # 滚动计算 ROE 的均值和标准差（20 天窗口）
    #     mean_roe = mean(self.roe, 20)  # 滚动均值
    #     std_roe = stddev(self.roe, 20)  # 滚动标准差

    #     # 计算当前 ROE 相对均值的偏离程度（标准化值）
    #     standardized_roe = (self.roe - mean_roe) / (std_roe + 1e-5)

    #     # 排名并返回因子值
    #     return rank(standardized_roe)
    
    # # 因子2：基于 ROA 的短期动量
    # def alpha_ROA_momentum(self):
    #     """
    #     alpha_ROA_momentum: 反映 ROA 的短期变化趋势
    #     逻辑：
    #     1. 计算 ROA 的短期变化（差分），捕捉盈利能力变化的方向。
    #     2. 结合当前 ROA 排名，强化动量的相对位置效应。
    #     3. 返回动量值的排名作为因子值。
    #     """
    #     # 计算 ROA 的 5 天变化
    #     roa_delta = delta(self.roa, 5)

    #     # 结合 ROA 当前排名计算动量值
    #     momentum = roa_delta * rank(self.roa)

    #     # 返回因子值
    #     return rank(momentum)
    
    # # 因子3：成交量与收盘价相关性
    # def alpha_Volume_Close(self):
    #     """
    #     alpha_Volume_Close: 捕捉成交量与价格的动态关系
    #     逻辑：
    #     1. 计算成交量与收盘价的滚动相关性（10 天窗口）。
    #     2. 高相关性表明成交量驱动价格波动，反映市场活跃度。
    #     3. 对相关性进行排名处理，返回因子值。
    #     """
    #     # 计算成交量与收盘价的滚动相关性
    #     corr_volume_close = corr(rank(self.volume), rank(self.close), 10)

    #     # 返回因子值
    #     return rank(corr_volume_close)
    
    # # 因子4：成交均价与收盘价差异
    # def alpha_Price_VWAP(self):
    #     """
    #     alpha_Price_VWAP: 衡量价格强弱的因子
    #     逻辑：
    #     1. 计算当前收盘价与成交均价的差异（相对比例）。
    #     2. 收盘价高于均价表明市场需求强，反之需求弱。
    #     3. 返回比例差的排名作为因子值。
    #     """
    #     # 计算收盘价与成交均价的差异
    #     price_strength = (self.close.shift(-1) - self.vwap.shift(-1) ) / (self.vwap.shift(-1) + 1e-5)

    #     # 返回因子值
    #     return rank(price_strength)

    # # 因子5：高价和低价的动态范围
    # def alpha_High_Low(self):
    #     """
    #     alpha_High_Low: 捕捉市场波动的因子
    #     逻辑：
    #     1. 计算高价和低价的差值，反映单日波动范围。
    #     2. 取差值的滚动均值（20 天窗口），衡量短期波动趋势。
    #     3. 返回均值的排名作为因子值。
    #     """
    #     # 计算高价和低价的差值
    #     high_low_spread = self.high - self.low

    #     # 滚动计算差值均值
    #     spread_mean = mean(high_low_spread, 20)

    #     # 返回因子值
    #     return rank(spread_mean)

    # # 因子6：基于季度销售增长率的波动性
    # def alpha_SGQ_volatility(self):
    #     """
    #     alpha_SGQ_volatility: 衡量季度销售增长的稳定性
    #     逻辑：
    #     1. 计算季度销售增长率的滚动标准差（20 天窗口）。
    #     2. 稳定的增长率通常与低风险和持续盈利能力相关。
    #     3. 返回波动率的排名作为因子值。
    #     """
    #     # 滚动计算季度销售增长率的标准差
    #     sgq_stddev = stddev(self.sgq, 20)

    #     # 返回因子值
    #     return rank(sgq_stddev)

    # # 因子7：基于季度利润率的动量
    # def alpha_PMQ_momentum(self):
    #     """
    #     alpha_PMQ_momentum: 捕捉季度利润率的变化趋势
    #     逻辑：
    #     1. 计算季度利润率的短期变化（10 天差分），反映利润率的动态。
    #     2. 结合季度利润率的排名，提升因子稳定性。
    #     3. 返回动量值的排名作为因子值。
    #     """
    #     # 计算季度利润率的短期变化
    #     pmq_delta = delta(self.pmq, 10)

    #     # 动量计算
    #     momentum = pmq_delta * rank(self.pmq)

    #     # 返回因子值
    #     return rank(momentum)

    # # 因子8：ROE 和 ROA 的综合盈利因子
    # def alpha_ROE_ROA(self):
    #     """
    #     alpha_ROE_ROA: 综合反映公司盈利能力的因子
    #     逻辑：
    #     1. 分别计算 ROE 和 ROA 的排名值。
    #     2. 将两者相乘，形成综合盈利能力信号。
    #     3. 返回综合信号的排名作为因子值。
    #     """
    #     # 计算 ROE 和 ROA 的排名
    #     roe_rank = rank(self.roe)
    #     roa_rank = rank(self.roa)

    #     # 综合盈利信号
    #     combined_profitability = roe_rank * roa_rank

    #     # 返回因子值
    #     return rank(combined_profitability)

    # # 因子9：市值因子
    # def alpha_size(self):
    #     """
    #     alpha_size: 市值因子，单位为千元人民币
    #     逻辑：
    #     1. 估算流通市值：成交均价（VWAP）乘以成交量。
    #     2. 使用对数化处理市值，减少极端值影响。
    #     3. 对对数市值进行排名，生成因子值。
    #     公式：
    #     市值 = max(成交均价 * 成交量, 1e-5)  # 防止为零的情况
    #     log_size = log(市值)  # 对市值取对数
    #     alpha_size = rank(log_size)  # 对数市值排名
    #     """
    #     market_cap = np.maximum(self.vwap * self.volume / 100000, 1e-5)  # 估算流通市值（单位：千元）
    #     log_size = np.log(market_cap)  # 对市值取对数
    #     return rank(log_size)  # 返回排名

    # # 因子10： 12-2 月动量因子 (Momentum Factor)
    # def alpha_momentum(self):
    #     """
    #     alpha_momentum: 动量因子
    #     逻辑：
    #     1. 计算过去 12 个月的累计收益（252 个交易日）。
    #     2. 减去过去 1 个月的累计收益（21 个交易日），去除短期影响。
    #     3. 对计算结果进行排名，生成因子值。
    #     公式：
    #     r12m = 过去 12 个月累计收益 = ts_sum(returns, 252)
    #     r1m = 过去 1 个月累计收益 = ts_sum(returns, 21)
    #     alpha_momentum = rank(r12m - r1m)  # 12个月动量减去1个月动量后排名
    #     """
    #     cumulative_return_12m = ts_sum(self.returns, 252)  # 计算过去 12 个月的累计收益
    #     cumulative_return_1m = ts_sum(self.returns, 21)  # 计算过去 1 个月的累计收益
    #     return rank(cumulative_return_12m - cumulative_return_1m)  # 计算并排名

    # # # 因子11：基于 CAPM 模型计算的 β 系数 (Beta Factor)
    # # def alpha_beta(self):
    # #     """
    # #     alpha_beta: CAPM β 系数
    # #     逻辑：
    # #     1. 对股票收益率与市场收益率进行线性回归。
    # #     2. 提取回归斜率 β 作为因子值。
    # #     3. 对 β 进行排名，生成因子值。
    # #     公式：
    # #     beta = regression_slope(returns, market_returns)  # 计算线性回归的斜率β
    # #     alpha_beta = rank(beta)  # 对 β 系数进行排名
    # #     """
    # #     beta = capm_beta(self.returns, self.market_return, self.rf)  # 计算 β 系数
    # #     return rank(beta)  # 对 β 进行排名

    # # 因子12：月度换手率因子 (Turnover Factor)
    # def alpha_turnover_month(self):
    #     """
    #     alpha_turnover: 月度换手率因子
    #     逻辑：
    #     1. 对换手率进行排名，生成因子值。
    #     公式：
    #     alpha_turnover = rank(turnover)  # 对月度换手率进行排名
    #     """
    #     return rank(self.turnover)  # 直接对月度换手率进行排名

    # # 因子13：短期反转因子 (Short-Term Reversal Factor)
    # def alpha_reversal(self):
    #     """
    #     alpha_reversal: 短期反转因子
    #     逻辑：
    #     1. 计算过去 1 个月（21 个交易日）的累计收益。
    #     2. 对累计收益取负值，捕捉短期反转效应。
    #     3. 对反转因子值进行排名，生成最终因子值。
    #     公式：
    #     srev_1m = -ts_sum(returns, 21)  # 过去1个月累计收益的负值
    #     alpha_reversal = rank(srev_1m)  # 对反转因子进行排名
    #     """
    #     monthly_return = ts_sum(self.returns, 21)  # 计算过去 1 个月的累计收益
    #     return rank(-monthly_return)  # 取负值并排名

    # # 因子14：流动性因子 (Liquidity Factor)
    # def alpha_liquidity(self):
    #     """
    #     alpha_liquidity: 流动性因子
    #     逻辑：
    #     1. 计算资产流动性指标，反映公司流动性水平。
    #     2. 对流动性指标进行排名，生成因子值。
    #     公式：
    #     alpha_liquidity = rank(资产流动性)  # 对资产流动性指标进行排名
    #     """
    #     return rank(self.alaq)  # 直接对资产流动性指标进行排名
    
    # # 因子15：基于 CTA 模型的现金占比因子 (Cash-to-Assets Factor)
    # def alpha_CTA(self):
    #     """
    #     alpha_CTA: 现金占比因子
    #     逻辑：
    #     1. 计算现金占比（CTA），反映公司现金流动性。
    #     2. 对现金占比进行排名，生成因子值。
    #     公式：
    #     alpha_CTA = rank(CTA)  # 对现金占比进行排名
    #     """
    #     return rank(self.cta)
    
    #     # Alpha#1	 (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)
    
    # # 因子16：alpha001
    # def alpha001(self):
    #     inner = self.close.copy()
    #     inner[self.returns < 0] = stddev(self.returns, 20)
    #     return rank(ts_argmax(inner ** 2, 5)) - 0.5
    
    # # 因子17：波动率偏度因子
    # def alpha_vol_skew(self):
    #     """
    #     alpha_vol_skew: 波动率偏度因子
    #     逻辑：
    #     1. 计算过去20天收益率的偏度，捕捉收益率分布的非对称性。
    #     2. 通过对偏度值进行排名，生成因子值。
    #     公式：
    #     skew = (E[(returns - mean)^3]) / stddev(returns)^3
    #     """
    #     vol_skew = skewness(self.returns, 20)
    #     return rank(vol_skew)

    # # 因子18：资金流动因子
    # def alpha_multi(self):
    #     """
    #     alpha_multi: 资金流动因子
    #     逻辑：
    #     1. 计算过去30天每日交易量与价格的乘积（近似资金流动）。
    #     2. 对这些值进行排名，生成因子值。
    #     公式：
    #     liquidity = ts_sum(volume * close, 30)
    #     alpha_multi = rank(liquidity)
    #     """
    #     liquidity = ts_sum(self.volume * self.close, 30)
    #     return rank(liquidity)
    
    # # 因子19：市场情绪因子
    # def alpha_sentiment(self):
    #     """
    #     alpha_sentiment: 市场情绪因子
    #     逻辑：
    #     1. 计算过去20天的收益率标准差，反映市场波动性。
    #     2. 对波动性进行排名，生成因子值。
    #     公式：
    #     alpha_sentiment = rank(stddev(returns, 20))
    #     """
    #     return rank(stddev(self.returns, 20))
    
    # # 因子20：收益率动量因子：过去60天的累计收益率
    # def alpha_momentum_60(self):
    #     """
    #     alpha_momentum: 收益率动量因子
    #     逻辑：
    #     1. 计算过去60天的累计收益率。
    #     2. 对累计收益率进行排名，捕捉长期收益趋势。
    #     公式：
    #     momentum = ts_sum(returns, 60)
    #     alpha_momentum = rank(momentum)
    #     """
    #     return rank(ts_sum(self.returns, 60))
    
    # # 因子21:交易密度因子
    # def alpha_trade_density(self):
    #     """
    #     alpha_trade_density: 交易密度因子
    #     逻辑：
    #     1. 计算过去20天的交易量均值。
    #     2. 对交易量均值进行排名，生成因子值。
    #     公式：
    #     alpha_trade_density = rank(mean(volume, 20))
    #     """
    #     return rank(mean(self.volume, 20))

    # # 因子22:十天换手率因子
    # def alpha_turnover_10(self):
    #     """
    #     alpha_turnover_10: 十天换手率因子
    #     逻辑：
    #     1. 计算过去10天的换手率。
    #     2. 对换手率进行排名，生成因子值。
    #     公式：
    #     alpha_turnover_10 = rank(turnover)
    #     """
    #     return rank(mean(self.turnover, 10))
    
    # # 因子23：财务健康因子
    # def alpha_financial_health(self):
    #     """
    #     alpha_financial_health: 财务健康因子
    #     逻辑：
    #     1. 计算现金占比（CTA）和流动资产占比（ALAQ）的加权平均值。
    #     2. 对健康因子值进行排名。
    #     """
    #     financial_health = 0.5 * self.cta + 0.5 * self.alaq
    #     return rank(financial_health)
    
    # # 因子24：相对强度因子
    # def alpha_rsi(self):
    #     """
    #     alpha_rsi: 相对强度因子
    #     逻辑：
    #     1. 计算过去14天价格的 RSI 指标。
    #     2. 通过排名生成因子值。
    #     """
    #     gains = np.maximum(self.returns, 0)
    #     losses = np.abs(np.minimum(self.returns, 0))
    #     avg_gain = mean(gains, 14)
    #     avg_loss = mean(losses, 14)
    #     rsi = 100 - (100 / (1 + avg_gain / avg_loss))
    #     return rank(rsi)
    
    # # 因子25：阿尔法市场因子
    # def alpha_market_alpha(self):
    #     """
    #     alpha_market_alpha: 阿尔法市场因子
    #     逻辑：
    #     1. 计算个股日收益率与市场收益率的差值。
    #     2. 对差值进行累计，并对因子值进行排名。
    #     """
    #     market_alpha = ts_sum(self.returns - self.market_return, 20)
    #     return rank(market_alpha)

    # 因子26：Rmrf 市场溢酬因子
    def alpha_Rmrf(self):
        """
        alpha_Rmrf: Rmrf 市场溢酬因子
        逻辑：
        r_m - r_f
        """
        # return rank(self.market_return - self.rf)
        return self.market_return - self.rf
    
    # # 因子27：Smb 市值因子
    # def alpha_Smb(self, weighted=weight):
    #     """
    #     alpha_Smb: Smb 市值因子
        
    #     逻辑：
    #     SMB_t = 1/3(SL_t + SM_t + SH_t) - 1/3(BL_t + BM_t + BH_t)
        
    #     根据流通市值（之后称为Size，股票价格P乘以在外流通股数Q）将股票按照中位数分为1：1的大市值（B）和小市值（S)股票；
        
    #     根据BM数据将股票分为3：4：3的高中低（H/M/L）三组；这样我们就有了2×3共计6种投资组合（SL/SM/SH/BL/BM/BH）。
        
    #     然后我们通过市值加权平均的方式求得各组的收益率
    #     """
    #     # 股票名
    #     stocks = self.returns.columns

    #     # 1. 按照市值分组：小市值 (S) 和 大市值 (B)
    #     median_size = self.size.median(axis=1)  # 每个时间点的市值中位数，按行计算
    #     small_cap = self.size.le(median_size, axis=0)  # 小市值股票，返回布尔矩阵
    #     large_cap = self.size.gt(median_size, axis=0)  # 大市值股票，返回布尔矩阵

    #     # 2. 根据 BM 值将股票分为三组：高账面市值比率（H），中账面市值比率（M），低账面市值比率（L）
    #     low_bm = self.bm.le(self.bm.quantile(1/3, axis=1), axis=0)  # 低账面市值比率（L）
    #     mid_bm = self.bm.gt(self.bm.quantile(1/3, axis=1), axis=0) & self.bm.le(self.bm.quantile(2/3, axis=1), axis=0)  # 中账面市值比率（M）
    #     high_bm = self.bm.gt(self.bm.quantile(2/3, axis=1), axis=0)  # 高账面市值比率（H）

    #     # 3. 计算每个组合的收益率
    #     SL = calculate_returns(self, small_cap & low_bm, weighted)
    #     SM = calculate_returns(self, small_cap & mid_bm, weighted)
    #     SH = calculate_returns(self, small_cap & high_bm, weighted)
    #     BL = calculate_returns(self, large_cap & low_bm, weighted)
    #     BM = calculate_returns(self, large_cap & mid_bm, weighted)
    #     BH = calculate_returns(self, large_cap & high_bm, weighted)

    #     # 4. 计算 SMB 因子
    #     SMB = (SL + SM + SH) / 3 - (BL + BM + BH) / 3

    #     results = pd.DataFrame(index=self.returns.index, columns=stocks)
    #     for stock in stocks:
    #         results[stock] = SMB
    #     return results
    
    # # 因子28：Hml 账面市值比因子
    # def alpha_Hml(self, weighted=weight):
    #     """
    #     alpha_Hml: 计算 HML 账面市值比因子
    #     逻辑：
    #     HML_t = 1/3(High_t) - 1/3(Low_t)
    #     """
    #     # 1. 根据 BM 值将股票分为三组：高账面市值比率（H），中账面市值比率（M），低账面市值比率（L）
    #     low_bm = self.bm.le(self.bm.quantile(1/3, axis=1), axis=0)  # 低账面市值比率（L）
    #     mid_bm = self.bm.gt(self.bm.quantile(1/3, axis=1), axis=0) & self.bm.le(self.bm.quantile(2/3, axis=1), axis=0)  # 中账面市值比率（M）
    #     high_bm = self.bm.gt(self.bm.quantile(2/3, axis=1), axis=0)
        
    #     High = calculate_returns(self, high_bm, weighted)
    #     Low = calculate_returns(self, low_bm, weighted)

    #     # 3. 计算HML因子
    #     HML = (High - Low)  # 计算高账面市值比率和低账面市值比率的回报差异
        
    #     stocks = self.returns.columns
    #     results = pd.DataFrame(index=self.returns.index, columns=stocks)
    #     for stock in stocks:
    #         results[stock] = HML
    #     return results
    
    # # # 因子30：账面市值比因子
    # # def alpha_bm(self):
    # #     """
    # #     alpha_bm: 账面市值比因子
    # #     逻辑：
    # #     1. 计算账面市值比因子 Hml 的排名值。
    # #     """
    # #     return rank(self.bm)
    
    # # # 因子31：换手率因子
    # # def alpha_turnover(self):
    # #     """
    # #     alpha_turnover: 前一天换手率因子
    # #     逻辑：
    # #     1. 计算换手率的排名值。
    # #     """
    # #     return rank(self.turnover.shift(-1))
    
    # # # 因子32：成交量波动率因子
    # # def alpha_CVD(self):
    # #     """
    # #     alpha_CVD: 成交量波动率因子
    # #     逻辑：
    # #     1. 计算成交量波动率的排名值。
    # #     """
    # #     return rank(self.cvd)
    
    # # # 因子33：每市值盈利因子
    # # def alpha_EPQ(self):
    # #     """
    # #     alpha_EPQ: 每市值盈利因子
    # #     逻辑：
    # #     1. 计算每市值盈利的排名值。
    # #     """
    # #     return rank(self.epq)
         
    # # # 因子34：企业价值倍数因子
    # # def alpha_EMQ(self):
    # #     """
    # #     alpha_EMQ: 企业价值倍数因子
    # #     逻辑：
    # #     1. 计算企业价值倍数的排名值。
    # #     """
    # #     return rank(self.emq)
    
    # # # 因子35：季度销售增长率因子
    # # def alpha_SGQ(self):
    # #     """
    # #     alpha_SGQ: 季度销售增长率因子
    # #     逻辑：
    # #     1. 计算季度销售增长率的排名值。
    # #     """
    # #     return rank(self.sgq)
    
    # # # 因子36：流动资产占比因子
    # # def alpha_ALAQ(self):
    # #     """
    # #     alpha_ALAQ: 流动资产占比因子
    # #     逻辑：
    # #     1. 计算流动资产占比的排名值。
    # #     """
    # #     return rank(self.alaq)
    
    # # # 因子37：季度利润率因子
    # # def alpha_PMQ(self):
    # #     """
    # #     alpha_PMQ: 季度利润率因子
    # #     逻辑：
    # #     1. 计算季度利润率的排名值。
    # #     """
    # #     return rank(self.pmq)
        
    # # # 因子38：Rmrf 市场溢酬因子
    # # def alpha_Rmrf(self):
    # #     """
    # #     alpha_Rmrf: Rmrf 市场溢酬因子
    # #     逻辑：
    # #     1. 计算市场溢酬因子 Rmrf 的排名值。
    # #     """
    # #     rmrf = self.market_return - self.rf
    # #     return rank(rmrf)
    
    # # # 因子39: ep 每股收益
    # # def alpha_EP(self):
    # #     """
    # #     alpha_EP: 每股收益因子
    # #     逻辑：
    # #     1. 计算每股收益的排名值。
    # #     """
    # #     return rank(self.ep)
    
    # 因子40：CH3
    # def alpha_CH3(self):
        # """
        # alpha_CH3: CH3 因子
        # 逻辑：
        # 1. 计算 CH3 因子的排名值。
        # """
        # self.extrareturn = self.market_return - self.rf
        
        # dates = self.returns.index
        # assets = self.returns.columns
        
        # size = pd.read_csv('alphas/ourAlphas/20110430/alpha_size.csv', index_col=0).reindex(index=dates, columns=assets)
        # ep = pd.read_csv('alphas/ourAlphas/20110430/alpha_EP.csv', index_col=0).reindex(index=dates, columns=assets)
        # turnover = pd.read_csv('alphas/ourAlphas/20110430/alpha_turnover.csv', index_col=0).reindex(index=dates, columns=assets)
        # Rmrf = pd.read_csv('alphas/ourAlphas/20110430/alpha_Rmrf.csv', index_col=0).reindex(index=dates, columns=assets)
        
        # y = self.returns - self.rf
        # pred = rollingCH3(size, ep, turnover, Rmrf, y)
        # return rank(pred)
  
    # # 因子41：CH3_Size
    # def alpha_CH3_Size(self, weighted=weight):
    #     """
    #     alpha_CH3_Size: CH3_Size 因子
    #     a. 将股票按市场规模中值拆分为两组：小股（S）和大股（B）。

    #     b. 将股票根据EP（earnings-price ratio）划分为三组：前30%为价值组（V）、中间40%为中间组（M）底部30%为成长组（G）

    #     c. 根据这些组的区间来形成六种最终Size-EP组合的价值加权组合：S/V、S/M、S/G、B/V、B/M和B/G。

    #     Size(规模因子)，SMB=1/3(S/V+S/M+S/G)–1/3(B/V+B/M+B/G)

    #     Value(价值因子），VMG=1/2(S/V+B/V)–1/2(S/G+B/G)

    #     Market(市场因子)=“价值加权组合的回报率”—“一年期存款利率”
    #     """
    #     # 股票名
    #     stocks = self.returns.columns

    #     # 1. 按照市值分组：小市值 (S) 和 大市值 (B)
    #     median_size = self.size.median(axis=1)  # 每个时间点的市值中位数，按行计算
    #     small_cap = self.size.le(median_size, axis=0)  # 小市值股票，返回布尔矩阵
    #     large_cap = self.size.gt(median_size, axis=0)  # 大市值股票，返回布尔矩阵

    #     # 2. 根据 EP（earnings-price ratio）划分为三组：价值（V）、中间（M）、成长（G）
    #     low_ep = self.ep.le(self.ep.quantile(1/3, axis=1), axis=0)  # 低EP（成长组 G）
    #     mid_ep = self.ep.gt(self.ep.quantile(1/3, axis=1), axis=0) & self.ep.le(self.ep.quantile(2/3, axis=1), axis=0)  # 中EP（中间组 M）
    #     high_ep = self.ep.gt(self.ep.quantile(2/3, axis=1), axis=0)  # 高EP（价值组 V）
        
    #     # 3. 计算每个组合的收益率
    #     S_V = calculate_returns(self, small_cap & high_ep, weighted)
    #     S_M = calculate_returns(self, small_cap & mid_ep, weighted)
    #     S_G = calculate_returns(self, small_cap & low_ep, weighted)
    #     B_V = calculate_returns(self, large_cap & high_ep, weighted)
    #     B_M = calculate_returns(self, large_cap & mid_ep, weighted)
    #     B_G = calculate_returns(self, large_cap & low_ep, weighted)
        
    #     # 4. 计算 SMB 因子
    #     SMB = (S_V + S_M + S_G) / 3 - (B_V + B_M + B_G) / 3
        
    #     results = pd.DataFrame(index=self.returns.index, columns=stocks)
    #     for stock in stocks:
    #         results[stock] = SMB
    #     return results
    
    # # 因子42：CH3_EP
    # def alpha_CH3_Value(self, weighted=weight):
    #     """
    #     alpha_CH3_Value: CH3_Value 因子
    #     """
    #             # 股票名
    #     stocks = self.returns.columns

    #     # 1. 按照市值分组：小市值 (S) 和 大市值 (B)
    #     median_size = self.size.median(axis=1)  # 每个时间点的市值中位数，按行计算
    #     small_cap = self.size.le(median_size, axis=0)  # 小市值股票，返回布尔矩阵
    #     large_cap = self.size.gt(median_size, axis=0)  # 大市值股票，返回布尔矩阵

    #     # 2. 根据 EP（earnings-price ratio）划分为三组：价值（V）、中间（M）、成长（G）
    #     low_ep = self.ep.le(self.ep.quantile(1/3, axis=1), axis=0)  # 低EP（成长组 G）
    #     mid_ep = self.ep.gt(self.ep.quantile(1/3, axis=1), axis=0) & self.ep.le(self.ep.quantile(2/3, axis=1), axis=0)  # 中EP（中间组 M）
    #     high_ep = self.ep.gt(self.ep.quantile(2/3, axis=1), axis=0)  # 高EP（价值组 V）
        
    #     # 3. 计算每个组合的收益率
    #     S_V = calculate_returns(self, small_cap & high_ep, weighted)
    #     S_M = calculate_returns(self, small_cap & mid_ep, weighted)
    #     S_G = calculate_returns(self, small_cap & low_ep, weighted)
    #     B_V = calculate_returns(self, large_cap & high_ep, weighted)
    #     B_M = calculate_returns(self, large_cap & mid_ep, weighted)
    #     B_G = calculate_returns(self, large_cap & low_ep, weighted)
        
    #     # 4. 计算 HML 因子
    #     HML = (S_V + B_V) / 2 - (S_G + B_G) / 2
        
    #     results = pd.DataFrame(index=self.returns.index, columns=stocks)
    #     for stock in stocks:
    #         results[stock] = HML
    #     return results
    
    # # 因子43：CH3_换手率
    # def alpha_CH3_turnover(self, weighted=weight):
    #     """
    #     alpha_turnover_factor: 构造基于换手率的积极/消极因子。
    #     - 前30%（积极组，高换手率）
    #     - 后30%（消极组，低换手率）
    #     """
    #     # 股票名
    #     stocks = self.returns.columns
        
    #     abnormal_turnover = ts_sum(self.turnover, 21) / ts_sum(self.turnover, 252)

    #     # 1. 计算换手率分位数
    #     top_30 = abnormal_turnover.gt(abnormal_turnover.quantile(0.7, axis=1), axis=0)  # 前30%（高换手率，积极组）
    #     bottom_30 = abnormal_turnover.le(abnormal_turnover.quantile(0.3, axis=1), axis=0)  # 后30%（低换手率，消极组）

    #     Opti = calculate_returns(self, top_30, weighted)
    #     Pessi = calculate_returns(self, bottom_30, weighted)
        
    #     # 3. 计算因子值
    #     PMO = Pessi - Opti
        
    #     results = pd.DataFrame(index=self.returns.index, columns=stocks)
        
    #     for stock in stocks:
    #         results[stock] = PMO

    #     return results
    
    # 生成Fama投资组合的被解释变量
    def alpha_portfolio_Fama(self, weighted=weight):
        """
        alpha_portfolio_Fama: 生成投资组合的被解释变量
        逻辑：
        按市值、账面市值比分成9组，以每一组的市值加权收益率/直接加权收益率作为被解释变量，对应的Rmrf, Smb, Hml(+constant)作为解释变量
        """
        # 股票名
        stocks = self.returns.columns
        
        # 1. 按市值分组：小市值 (S)、中市值 (M)、大市值 (B)
        size_quantiles = self.size.quantile([1/3, 2/3], axis=1)
        small_cap = self.size.le(size_quantiles.iloc[0], axis=0)  # 小市值股票
        mid_cap = self.size.gt(size_quantiles.iloc[0], axis=0) & self.size.le(size_quantiles.iloc[1], axis=0)  # 中市值股票
        large_cap = self.size.gt(size_quantiles.iloc[1], axis=0)  # 大市值股票
        
        # 2. 根据账面市值比（B/M）划分为三组：低账面市值比（L）、中账面市值比（M）、高账面市值比（H）
        bm_quantiles = self.bm.quantile([1/3, 2/3], axis=1)
        low_bm = self.bm.le(bm_quantiles.iloc[0], axis=0)  # 低账面市值比（L）
        mid_bm = self.bm.gt(bm_quantiles.iloc[0], axis=0) & self.bm.le(bm_quantiles.iloc[1], axis=0)  # 中账面市值比（M）
        high_bm = self.bm.gt(bm_quantiles.iloc[1], axis=0)  # 高账面市值比（H）
        
        # 3. 计算每个组合的加权收益率（市值加权收益率）
        S_L = calculate_returns(self, small_cap & low_bm, weighted)
        S_M = calculate_returns(self, small_cap & mid_bm, weighted)
        S_H = calculate_returns(self, small_cap & high_bm, weighted)
        M_L = calculate_returns(self, mid_cap & low_bm, weighted)
        M_M = calculate_returns(self, mid_cap & mid_bm, weighted)
        M_H = calculate_returns(self, mid_cap & high_bm, weighted)
        B_L = calculate_returns(self, large_cap & low_bm, weighted)
        B_M = calculate_returns(self, large_cap & mid_bm, weighted)
        B_H = calculate_returns(self, large_cap & high_bm, weighted)
        
        # 创建一个 DataFrame 来存储每个股票的被解释变量
        names = ['S_L', 'S_M', 'S_H', 'M_L', 'M_M', 'M_H', 'B_L', 'B_M', 'B_H']
        results = pd.DataFrame(index=self.returns.index, columns=names)
        
        for name in names:
            results[name] = eval(name)
            
        results.to_csv(f'alphas/multialpha/alpha_portfolio_Fama_{weight}.csv')
    
    # 生成CH3投资组合的被解释变量
    def alpha_portfolio_CH3(self, weighted=weight):
        """
        alpha_portfolio_CH3: 生成投资组合的被解释变量
        逻辑：
        按市值、账面市值比、换手率分成8组(2 * 2 * 2)，以每一组的市值加权收益率/直接加权收益率作为被解释变量，对应的CH3, CH3_Size, CH3_turnover(+constant)作为解释变量
        """
        # 股票名
        stocks = self.returns.columns
        
        # 1. 按市值分组：小市值 (S)、大市值 (B)
        size_quantiles = self.size.quantile(1/2, axis=1)
        small_cap = self.size.le(size_quantiles, axis=0)
        large_cap = self.size.gt(size_quantiles, axis=0)
        
        # 2. 根据账面市值比（B/M）划分为两组：低账面市值比（L）、高账面市值比（H）
        bm_quantiles = self.bm.quantile(1/2, axis=1)
        low_bm = self.bm.le(bm_quantiles, axis=0)
        high_bm = self.bm.gt(bm_quantiles, axis=0)
        
        # 3. 根据换手率划分为两组：低换手率（L）、高换手率（H）
        turnover_quantiles = self.turnover.quantile(1/2, axis=1)
        low_turnover = self.turnover.le(turnover_quantiles, axis=0)
        high_turnover = self.turnover.gt(turnover_quantiles, axis=0)
        
        # 5. 计算每个股票的被解释变量（市值加权收益率 或 直接加权收益率）
        S_L_L = calculate_returns(self, small_cap & low_bm & low_turnover, weighted)
        S_L_H = calculate_returns(self, small_cap & low_bm & high_turnover, weighted)
        S_H_L = calculate_returns(self, small_cap & high_bm & low_turnover, weighted)
        S_H_H = calculate_returns(self, small_cap & high_bm & high_turnover, weighted)
        B_L_L = calculate_returns(self, large_cap & low_bm & low_turnover, weighted)
        B_L_H = calculate_returns(self, large_cap & low_bm & high_turnover, weighted)
        B_H_L = calculate_returns(self, large_cap & high_bm & low_turnover, weighted)
        B_H_H = calculate_returns(self, large_cap & high_bm & high_turnover, weighted)

        # 创建一个 DataFrame 来存储每个股票的被解释变量
        names = ['S_L_L', 'S_L_H', 'S_H_L', 'S_H_H', 'B_L_L', 'B_L_H', 'B_H_L', 'B_H_H']
        results = pd.DataFrame(index=self.returns.index, columns=names)
        
        for name in names:
            results[name] = eval(name)
            
        results.to_csv(f'alphas/multialpha/alpha_portfolio_CH3_{weight}.csv')

    
    
if __name__ == '__main__':
    year = '2011'
    start_date = '20110430'
    end_date = '20230430'
    
    list_assets,df_asserts = get_hs300_stocks(f"{year}-01-01")

    ################ 计算所有 #################   
    ourAlphas.generate_alphas(start_date, end_date, list_assets,"sh000300")
    
    # ret = ourAlphas.generate_alpha_single('alpha001', year, list_assets, "sh000300", True)
    # print(ret)