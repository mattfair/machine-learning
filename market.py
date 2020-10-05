# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def Brownian(seed, N):
    np.random.seed(seed)
    dt = 1./N                                    # time step
    b = np.random.normal(0., 1., int(N))*np.sqrt(dt)  # brownian increments
    W = np.cumsum(b)                             # brownian path
    return W, b

# Parameters
#
# So:     initial stock price
# mu:     returns (drift coefficient)
# sigma:  volatility (diffusion coefficient)
# W:      brownian motion
# T:      time period
# N:      number of increments
def GBM(So, mu, sigma, W, T, N):
    t = np.linspace(0., 1., N)
    S = []
    S.append(So)
    for i in range(1, int(N)):
        drift = (mu - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i-1]
        S_temp = So*np.exp(drift + diffusion)
        S.append(S_temp)
    return S, t


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def ohlc(values):
    if len(values) == 0:
        return None
    #print (values)
    o = h = l = c = values[0]
    for v in values:
        h = max(h, v)
        l = min(l, v)
        c = v
    if l > h:
        raise ValueError("invalid ohlc data: low above high")
    return (o, h, l, c)

def generate_market_data(N, startDate):
    so = np.random.uniform(-1000, 10000)
    mu = np.random.uniform(0.0, 0.15)
    sigma = np.random.uniform(0.0, 1.0)
    T = 1.

    # N in days, convert to minutes to generate
    # calculate 24 hours and then remove overnight data
    overnight_minutes = 60*2
    minutes_per_day = 24*60
    minutes = N*minutes_per_day
    W = Brownian(np.random.seed(), minutes)[0]
    brownian = GBM(so, mu, sigma, W, T, minutes)[0]

    df = pd.DataFrame(index=pd.date_range(start=startDate, periods=N))
    # daily browniand motion
    brownian_bars = [bar for bar in chunks(brownian, minutes_per_day)]

    # calculate bars
    ohlc_bars = []
    i = 0
    for bar in brownian_bars:
        # chop off overnight time period where there is no trading
        if len(bar) > overnight_minutes:
            (o, h, l, c) = ohlc(bar[:-overnight_minutes])
            dti = df.index[i]
            df.at[dti, 'open'] = o
            df.at[dti, 'high'] = h
            df.at[dti, 'low'] = l
            df.at[dti, 'close'] = c
            df.at[dti, 'volume'] = 0
        else:
            print("skipping bar, does not have enough bars:", len(bar))
        i += 1

    return df.dropna()

def normalize_ohlc(df):
    s = pd.concat([df.open, df.high, df.low, df.close])
    min = s.min()
    max = s.max()
    diff = max-min

    if diff == 0:
        return None
    else:
        df.open = (df.open - min)/diff
        df.high = (df.high - min)/diff
        df.low = (df.low - min)/diff
        df.close = (df.close - min)/diff
        return df
