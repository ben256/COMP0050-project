import logging
import os

import pandas as pd
import yfinance
from matplotlib import pyplot as plt


def check_data(
        data_path: str,
        save_path: str,
        sector_stock_count: int,
        total_count: int = None,
        period: int = 10,  # Years
        interval: str = '1d',
        start_date: str = '2015-02-09',
        source = None,
        allow_missing: bool = False,
):
    """
    Notes:
    1. Sector stock count must be met for acceptance
    2. Total count only applicable if sector_stock_count is not present
    3. Period must be met for acceptance
    4. No source means all sources are accepted, with priority give to higher market cap stocks

    :param data_path:
    :param save_path:
    :param sector_stock_count:
    :param total_count:
    :param period:
    :param interval:
    :param source:
    :param allow_missing:
    :return bool:
    """

    accept_data = True
    selected_stocks = []

    price_data = pd.read_csv(save_path, index_col='Date', parse_dates=True)
    stock_info = pd.DataFrame()

    # Remove columns with all missing data
    price_data.dropna(axis=1, how='all', inplace=True)

    # Filter by source
    if source:
        for exchange in source:
            exchange_info = pd.read_csv(f'{data_path}/{exchange}.csv')
            exchange_info['Exchange'] = exchange
            stock_info = pd.concat([stock_info, exchange_info])
    else:
        nasdaq_info = pd.read_csv(f'{data_path}/nasdaq.csv')
        nyse_info = pd.read_csv(f'{data_path}/nyse.csv')

        nasdaq_info['Exchange'] = 'nasdaq'
        nyse_info['Exchange'] = 'nyse'

        stock_info = pd.concat([nasdaq_info, nyse_info])

    # Drop miscellaneous sector
    stock_info = stock_info[stock_info['Sector'] != 'Miscellaneous']

    # Sort by market cap
    stock_info.sort_values(['Market Cap'], ascending=False, inplace=True)

    start_date_converted = pd.Timestamp(start_date, tz='America/New_York')
    end_date = start_date_converted + pd.offsets.DateOffset(years=period)

    if not sector_stock_count:
        ...

    else:
        sector_groups = stock_info.groupby('Sector')

        try:
            for sector_name, sector_data in sector_groups:
                sector_count = 0

                for index, row in sector_data.iterrows():
                    ticker = row['Symbol'].strip()

                    if ticker.lower() not in price_data.columns:
                        try:
                            ticker_data = yfinance.Ticker(ticker).history(period=f'{period}y', interval=interval, start=start_date, raise_errors=True)
                            logging.info(f'Fetched data for {ticker}')
                            price_data[ticker.lower()] = ticker_data['Close']

                            selected_stocks.append(ticker)
                            sector_count += 1

                        except Exception as e:
                            logging.warning(f'Failed to fetch data for {ticker}, switching to max period')
                            ticker_data = yfinance.Ticker(ticker).history(period='max', interval=interval)

                            if ticker_data.empty:
                                logging.warning(f'Failed to fetch data for {ticker}, skipping')
                                continue

                            if allow_missing:
                                logging.warning(f'Allowing missing data for {ticker}')
                                price_data[ticker.lower()] = ticker_data['Close']

                                selected_stocks.append(ticker)
                                sector_count += 1

                    else:
                        ticker_data = price_data[[ticker.lower()]].loc[start_date_converted:end_date]
                        if ticker_data.isnull().values.any():
                            ticker_start_date = ticker_data.dropna().index[0].strftime('%Y-%m-%d')
                            logging.debug(f'Missing data for {ticker}, data start date: {ticker_start_date}')
                            if allow_missing:
                                logging.warning(f'Allowing missing data for {ticker}')
                                selected_stocks.append(ticker)
                                sector_count += 1

                        else:
                            logging.debug(f'Found data for {ticker}')
                            selected_stocks.append(ticker)
                            sector_count += 1

                    if sector_count == sector_stock_count:
                        logging.info(f'Found {sector_count} stocks for sector {sector_name}')
                        # print(f'Current list of selected stocks: {selected_stocks}')
                        # print(f'Len {len(selected_stocks)}')
                        break

                # Failed to get enough stock data for sector
                if sector_count < sector_stock_count:
                    logging.warning(f'Only found {sector_count} stocks for sector {sector_name}, expected {sector_stock_count}')
                    accept_data = False

        except Exception as e:
            logging.error(f'Unexpected error: {e}')
            logging.info('Saving price data')
            price_data.to_csv(save_path)

            raise e

    price_data.to_csv(save_path)

    filtered_stock_info = stock_info[stock_info['Symbol'].isin(selected_stocks)]
    total_stock_count = len(filtered_stock_info)
    sector_counts = filtered_stock_info['Sector'].value_counts()

    logging.info(f'Total stock count: {total_stock_count}')
    logging.info(f'Sector counts:\n{sector_counts}')

    return accept_data, filtered_stock_info


def fetch_data(
        sector_stock_count: int = 50,
        total_count: int = None,
        source = None,
        data_path: str = './data',
        save_path: str = './data/processed_data.csv',
        period: int = 10,  # Years
        interval: str = '1d',
        start_date: str = '2015-02-09',
        allow_missing: bool = False,
        fill_missing: bool = False,
        raise_errors: bool = True,
):
    if source is None:
        source = ['nasdaq', 'nyse']

    if os.path.isfile(save_path):
        logging.info('Data already exists at save path, checking completeness')

        accept_data, selected_stocks = check_data(
            data_path=data_path,
            save_path=save_path,
            sector_stock_count=sector_stock_count,
            total_count=total_count,
            period=period,
            interval=interval,
            start_date=start_date,
            source=source,
            allow_missing=allow_missing,
        )

        selected_stocks_symbols = selected_stocks['Symbol'].str.lower().values
        price_data = pd.read_csv(save_path, index_col='Date', parse_dates=True)
        filtered_price_data = price_data[selected_stocks_symbols]

        if accept_data:
            logging.info('Data is complete')
            return filtered_price_data, selected_stocks
        else:
            if not raise_errors:
                logging.warning('Errors suppressed, returning data')
                return filtered_price_data, selected_stocks
            else:
                logging.error('Data is incomplete, raise errors is set to True')
                raise Exception('Data is incomplete')

    else:
        logging.warning('Data not found at save path, fetching new data')


def plot_missing(price_data, stock_info, sector=None):

    if sector:
        sector_stocks = stock_info[stock_info['Sector'] == sector]['Symbol']
        price_data = price_data[sector_stocks.str.lower()]

    fig, ax = plt.subplots(figsize=(20, 16), tight_layout=True)

    for index, row in enumerate(price_data.clip(0, 0).values.T):
        ax.plot(price_data.index, row + index + 1, linewidth='10')

    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor', linestyle='-', alpha=0.3)
    plt.minorticks_on()
    plt.legend()
    plt.show()
    plt.close()


def create_mapping(timeseries_data):
    ...
