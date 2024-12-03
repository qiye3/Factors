import pandas as pd
import numpy as np
from functools import wraps
from scipy.stats import rankdata
from numpy import abs
from numpy import log
from numpy import sign
import statsmodels.api as sm
from tqdm import tqdm

def returns(df):
    return df.rolling(2).apply(lambda x: x.iloc[-1] / x.iloc[0]) - 1

def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    
    return df.rolling(window).sum()

def sma(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()

def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()

def mean(df, window=10):
    """
    Wrapper function to estimate rolling mean.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()

def corr(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y).fillna(0).replace([np.inf, -np.inf], 0)

def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)

def skewness(df, window=10):
    """
    Wrapper function to estimate rolling skewness.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).skew()

def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na,method='min')[-1]

def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)

def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)

def product(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)

def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()

def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()

def delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)

def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)

def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    return df.rank(axis=1, method='min', pct=True).fillna(0)
    # return df.rank(pct=True)

def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())

def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1 

def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1

def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    weights = np.array(range(1, period+1))
    sum_weights = np.sum(weights)
    return df.rolling(period).apply(lambda x: np.sum(weights*x) / sum_weights)

def max(sr1,sr2):
    return np.maximum(sr1, sr2)

def min(sr1,sr2):
    return np.minimum(sr1, sr2)

def regression_slope(y, X):
    """
    使用 statsmodels 计算回归斜率（系数）beta。
    
    参数：
        y (pd.Series): 因变量。
        X (pd.DataFrame): 自变量矩阵。
    
    返回：
        pd.Series: 回归系数 beta，包含每个变量的系数。
    """
    # 在 X 中添加截距列（常数项）
    X = sm.add_constant(X)  # 自动添加截距项
    
    # 使用 statsmodels 的 OLS 模型进行回归
    model = sm.OLS(y, X).fit()
    
    # 返回回归系数，结果是一个 pd.Series 类型
    return model.params

def regression_residual(y, X):
    """
    计算回归残差。
    
    参数：
        y (np.ndarray): 因变量（一维数组）。
        X (np.ndarray): 自变量矩阵（二维数组，包含截距列）。
    
    返回：
        np.ndarray: 残差值。
    """
    # 计算回归系数
    beta = regression_slope(y, X)

    # 添加截距列（如果 X 不包含）
    if X.ndim == 1:
        X = X[:, np.newaxis]
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # 添加截距列

    # 计算预测值
    y_pred = X @ beta

    # 计算残差
    residuals = y - y_pred
    return residuals

def generate_all_dates(start_date, end_date):
    """
    生成从 start_date 到 end_date 的所有交易日日期列表。
    
    参数：
        start_date (str): 开始日期。
        end_date (str): 结束日期。
    
    返回：
        pd.DatetimeIndex: 日期列表。
    """
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    return all_dates

def rollingCH3(size, ep, turnover, mkt, returns, window=252):
    """
    计算 CH3 三因子模型的滚动 beta。
    可优化：可以使用rolling OLS来计算滚动beta。
    
    参数:
        X (DataFrame): 因子数据，列包括 size, ep, turnover, extrareturn。
        returns (DataFrame): 资产收益数据。
        window (int): 滚动窗口大小。
        
    返回:
        预测值 DataFrame，列为资产代码，行为日期。
    """
    assets = returns.columns
    dates = returns.index
    preds = pd.DataFrame(index=dates, columns=assets)

    # 滚动窗口回归
    for asset in tqdm(assets):
        for i in range(window, len(dates)):
            # 提取滚动窗口数据
            window_returns = returns.iloc[i-window:i, :][asset]
            window_factors = pd.concat([size.iloc[i-window:i][asset],ep.iloc[i-window:i][asset],turnover.iloc[i-window:i][asset], mkt.iloc[i-window:i][asset]],axis=1)
            window_factors.columns = ['size', 'ep', 'turnover', 'mkt']

            # 清理数据：去除 NaN 和 Inf
            combined = pd.concat([window_returns, window_factors], axis=1).dropna()
            if combined.empty:
                continue

            window_returns_clean = combined.iloc[:, 0]
            window_factors_clean = combined.iloc[:, 1:]

            # 添加常数项
            factors_with_const = sm.add_constant(window_factors_clean)

            # 回归模型
            model = sm.OLS(window_returns_clean, factors_with_const).fit()
            
            if model.params.shape[0] == 5:
                preds.loc[dates[i], asset] = model.predict([1, size.iloc[i][asset], ep.iloc[i][asset], turnover.iloc[i][asset], mkt.iloc[i][asset]])[0]
                
            else:
                preds.loc[dates[i], asset] = model.predict([size.iloc[i][asset], ep.iloc[i][asset], turnover.iloc[i][asset], mkt.iloc[i][asset]])[0]
            
    return preds

def rolling_ols(y, X, window):
    """
    滚动 OLS 回归函数，先对每个窗口内的缺失值进行删除（dropna），然后进行 OLS 回归。
    返回预测值（即预测的 y 值）。
    """
    result = y.rolling(window).apply(
        lambda y_window: sm.OLS(
            y_window.dropna(),  # 删除 y 中的缺失值
            sm.add_constant(X.loc[y_window.index].dropna())  # 删除 X 中的缺失值
        ).fit().predict(sm.add_constant(X.loc[y_window.index].dropna())) 
        if y_window.dropna().notnull().all() and X.loc[y_window.index].dropna().notnull().all()
        else np.nan
    )
    return result

def capm_beta(returns, market_returns, rf_rate, window=252):
    """
    使用 rolling_ols 函数计算 CAPM β 系数。
    
    参数：
        returns (pd.DataFrame): 股票收益率，列为股票代码，行为日期。
        market_returns (pd.Series): 市场收益率（超额收益率）。
        rf_rate (pd.Series): 无风险利率。
        window (int): 滚动窗口大小。
        
    返回：
        beta_factors (pd.DataFrame): 每只股票的滚动 β 系数，列为股票代码，行索引为日期。
    """
    rf_rate = pd.Series(rf_rate.iloc[:][0], index=returns.index)  # 无风险利率
    # 股票超额收益率
    excess_returns = returns.sub(rf_rate, axis=0)  # 每只股票的超额收益率
    market_excess = market_returns - rf_rate       # 市场的超额收益率
    
    # 初始化结果表
    beta_factors = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    # 逐列计算每只股票的滚动 β 系数
    for stock in returns.columns:
        stock_returns = excess_returns[stock]
        # 调用 rolling_ols
        betas = rolling_ols(y=stock_returns, X=market_excess, window=window)
        beta_factors[stock] = betas
    
    return beta_factors

# 根据股票代码判断交易所
def identify_exchange(code):
    if str(code).startswith(('600', '601', '603', '688')):
        return 0  # 上交所
    elif str(code).startswith(('000', '002', '300')):
        return 1  # 深交所
    else:
        return 2  # 其他交易所
