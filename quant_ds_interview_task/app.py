%matplotlib qt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import seaborn as sns

from pylab import rcParams
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn import metrics

def remove_outliers(df, sn, cutoff=5):
    """
    Identifies outliers from standardized values and returns dataframe with no
    outliers.
    
    Parameters
    ----------
    df : pd.DataFrame
        raw financial time series.
    sn : pd.DataFrame
        standard normalized log returns time series.
    cutoff : float, optional
        standard deviation cut-off for identifying outliers. The default is 5.

    Returns
    -------
    masked : pd.DataFrame
        masked and ffilled dataframe containing no outliers.

    """
    cond = sn.abs() >= cutoff
    masked = df.mask(cond).ffill()
    return masked

def load_and_describe():
    df = pd.read_csv(r"C:\Users\Mark\Documents\Job hunting stuff\quant-ds-interview-task\quant-ds-interview-master\tests\data\currency_pair_prices.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    print(df.describe())
    corr = df.corr().sort_values(by='EUR/USD', ascending=False)
    ax = sns.heatmap(corr, cmap="YlGnBu", annot=True, 
           fmt='.2f',)
    return df

def remove_nan(df):
    df = df.ffill()    ## assumption that missing data due to non-trading therefore price stays the same
    return df

def calc_log_returns(df):
    log_returns = np.log(1 + df.pct_change()).iloc[1:]
    log_returns.plot()
    return log_returns

def normalise(log_returns):
    sn = (log_returns - log_returns.mean()) / log_returns.std()    
    #sn.plot(kind='hist', bins=20)
    #sn.plot(kind='hist', bins=10, logy=True)
    fig, ax = plt.subplots()
    sns.histplot(sn.melt(), x='value', hue='variable',
             multiple='dodge', shrink=.75, bins=20, log_scale=(False, True) );
    pd.plotting.scatter_matrix(sn, diagonal='kde', figsize=(10, 10));
    return sn

def seasonal_decomposition(df):
    rcParams['figure.figsize'] = 18, 8
    #decomposition = sm.tsa.seasonal_decompose(log_returns['EUR/USD'], model='additive', period = 20)
    decomposition = sm.tsa.seasonal_decompose(df['EUR/USD'], model='additive', period = 20)
    fig = decomposition.plot()
    plt.show()

def simple_arma_model(log_returns, currency = 'EUR/USD'):
    split_date = log_returns.index[-78]  # ts is length 784
    
    train = log_returns.loc[log_returns.index < pd.to_datetime(split_date)][currency]
    test = log_returns.loc[log_returns.index >= pd.to_datetime(split_date)][currency]
    
    model = ARIMA(train, order=(1, 0, 5)); # log returns are already cointegrated
    results = model.fit();
    arma_prediction = results.predict(
        start=split_date, end=test.index[-1], dynamic=False);
    plt.figure(figsize=(10, 5))
    l1, = plt.plot(log_returns[currency], label='Observation')
    l2, = plt.plot(arma_prediction, label='ARMA(1,1)')
    plt.legend(handles=[l1, l2])
    plt.savefig('ARMA(1,1) prediction', bbox_inches='tight', transparent=False)    
    print(f'ARMA RMSE is : {np.sqrt(metrics.mean_squared_error(test, arma_prediction))}')
    return arma_prediction
    
    
def exog_arma_model(log_returns, currency = 'AUD/USD', exog_currency = 'EUR/USD'):
    split_date = log_returns.index[-78]  # ts is length 784
    
    train = log_returns.loc[log_returns.index < pd.to_datetime(split_date)][[currency, exog_currency]]
    test = log_returns.loc[log_returns.index >= pd.to_datetime(split_date)][[currency, exog_currency]]
    model = ARIMA(train[currency], exog=train[exog_currency], order=(1, 0, 1)); # log returns are already cointegrated
    results = model.fit();
    armax_prediction = results.predict(
        start=split_date, end=test.index[-1], dynamic=False, exog=test[exog_currency] );
    plt.figure(figsize=(10, 5))
    l1, = plt.plot(log_returns[currency], label='Observation')
    l2, = plt.plot(armax_prediction, label='ARMAX(1,1)')
    plt.legend(handles=[l1, l2])
    plt.savefig('ARMAX(1,1) prediction', bbox_inches='tight', transparent=False)   
    print(f'ARMAX RMSE is : {np.sqrt(metrics.mean_squared_error(test[currency], armax_prediction))}')
    return armax_prediction
    
def plot_acf_pacf(log_returns, currency = 'EUR/USD'):
    acf(log_returns[currency])
    lags = 5
    plot_acf(log_returns[currency], lags = lags, c = 'g',);

    pacf(log_returns[currency]);
    lags = 5
    plot_pacf(log_returns[currency], lags = lags, c = 'g', method='ywm');
    
def run():
    """
    Entry-point for app
    """
    df = load_and_describe()
    df = remove_nan(df)
    log_returns = calc_log_returns(df)
    sn = normalise(log_returns)    
    log_returns = remove_outliers(log_returns, sn)
    seasonal_decomposition(df)
    seasonal_decomposition(log_returns)
    plot_acf_pacf(log_returns,) #currency='AUD/USD')
    simple_arma_model(log_returns)
    simple_arma_model(log_returns, currency='AUD/USD')
    exog_arma_model(log_returns)
    return df


if __name__ == "__main__":
    out = run()