from annealing import simulated_annealing_ordering
from correlation import *
from data_processing import *
from graphs import *
from utils import *


def run_group_correlation(
        N_g = 9,
        initial_temperature = 0.5,
        cooling_rate = 0.9997,
        cut_off = 0.1,
):

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

    log_returns = compute_log_returns(prices)
    correlation_matrix = compute_correlation_matrix(log_returns)
    corr_eigenvalues, corr_eigenvectors = compute_eigenvalues(correlation_matrix)

    C_g = compute_group_modes(corr_eigenvalues, corr_eigenvectors, N_g)
    best_order, best_energy, iteration_count, energy_history = simulated_annealing_ordering(
        C_g,
        cutoff=cut_off,
        initial_temp=initial_temperature,
        cooling_rate=cooling_rate,
        iterations=300000,
        tol=1e-2,
        patience=100000,
        return_history=True,
        individual_logging=True,
    )

    plot_heat_map(C_g, best_order, output_folder)
    plot_energy_history(energy_history, output_folder)
