import numpy as np


def compute_log_returns(price_data, as_dataframe=False):
    returns = np.log(price_data / price_data.shift(1))
    returns.dropna(inplace=True)
    if as_dataframe:
        return returns
    else:
        return returns.values
