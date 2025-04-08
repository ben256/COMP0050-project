import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import SparsePCA

from lib.correlation import compute_correlation_matrix
from lib.data_processing import fetch_data
from lib.utils import create_output_folder, compute_log_returns


def run_sparse_pca():

    output_folder = create_output_folder('./output', 'graphing')

    prices, stock_info = fetch_data(
        sector_stock_count=50,
        total_count=None,
        source=['nasdaq', 'nyse'],
        data_path='./data',
        save_path='./data/processed_data.csv',
        period=10,  # Years
        interval='1d',
        start_date='2015-02-09',
        allow_missing=False,
        fill_missing=False,
        raise_errors=True,
    )

    target_sectors = ['Energy', 'Technology', 'Telecommunications']
    stock_info = stock_info[stock_info['Sector'].isin(target_sectors)]

    for sector, sector_data in stock_info.groupby('Sector'):
        sector_stocks = sector_data['Symbol'].str.lower().to_list()
        sector_prices = prices[sector_stocks]

        log_returns = compute_log_returns(sector_prices)

        correlation = compute_correlation_matrix(log_returns)
        correlation_df = pd.DataFrame(correlation, index=sector_stocks, columns=sector_stocks)

        n_sparse_components = 2
        alpha = 0.8  # Larger alpha => more sparsity

        spca = SparsePCA(n_components=n_sparse_components, alpha=alpha, random_state=42, max_iter=1000)
        spca.fit(correlation_df)

        sparse_components = spca.components_
        component_df = pd.DataFrame(
            sparse_components,
            index=[f"Comp {i+1}" for i in range(n_sparse_components)],
            columns=sector_stocks
        )

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            component_df,
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f',
            annot_kws={'rotation': 90}
        )
        plt.title(f"Sparse PCA Components Heatmap (Sector: {sector})")
        plt.xlabel("Stocks")
        plt.ylabel("Sparse Components")
        plt.tight_layout()
        plt.savefig(f'{output_folder}/sparse_pca_{sector}.png', dpi=300)
        plt.show()
        plt.close()


if __name__ == '__main__':
    run_sparse_pca()
