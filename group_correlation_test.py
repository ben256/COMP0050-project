import json

from lib.annealing import simulated_annealing_ordering
from lib.correlation import *
from lib.data_processing import *
from lib.graphs import *
from lib.utils import *


def run_group_correlation(
        N_g = 19,
        initial_temperature = 1.0,
        cooling_rate = 0.9997,
        cut_off = 0.09,
):
    np.random.seed(42)

    output_folder = create_output_folder('./output', 'graphing')
    # output_folder = './output/temp'

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

    stocks_map = {index: stock for index, stock in enumerate(stock_info['Symbol'].str.lower().to_list())}
    sector_map = {index: sector for index, sector in enumerate(stock_info['Sector'].to_list())}

    log_returns = compute_log_returns(prices)
    correlation_matrix = compute_correlation_matrix(log_returns)
    corr_eigenvalues, corr_eigenvectors = compute_eigenvalues(correlation_matrix)

    C_g = compute_group_modes(corr_eigenvalues, corr_eigenvectors, N_g)
    best_order, best_energy, iteration_count, energy_history = simulated_annealing_ordering(
        C_g,
        cutoff=cut_off,
        initial_temp=initial_temperature,
        cooling_rate=cooling_rate,
        iterations=10000000,
        tol=1e-3,
        patience=10000,
        return_history=True,
        individual_logging=True,
    )

    output = {
        'best_energy': best_energy,
        'iteration': iteration_count,
        'N_g': N_g,
        'initial_temperature': initial_temperature,
        'cooling_rate': cooling_rate,
        'cut_off': cut_off,
        'best_order': best_order,
        'energy_history': energy_history,
        'stocks_map': stocks_map,
        'sector_map': sector_map,
    }

    with open(f'{output_folder}/parameter_selection_output.json', 'w') as f:
        json.dump(output, f)

    plot_heat_map(C_g, best_order, stocks_map, sector_map, output_folder)
    plot_energy_history(energy_history, output_folder)


if __name__ == '__main__':
    run_group_correlation()
