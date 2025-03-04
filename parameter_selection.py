import itertools
import json
import logging

from tqdm import tqdm

from annealing import simulated_annealing_ordering
from correlation import *
from data_processing import *
from utils import *


def run_parameter_selection():

    tuning_folder = create_output_folder('./output', 'tuning')

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

    N_g_values = [7, 8, 9, 10, 11, 12, 13]
    intial_temperatures = [0.2, 0.4, 0.6, 0.8, 1.0]
    cooling_rates = [0.9995, 0.9999, 0.99999]
    cut_offs = [0.1, 0.15, 0.2]

    ranges = {
        "N_g": N_g_values,
        "initial_temperature": intial_temperatures,
        "cooling_rate": cooling_rates,
        "cut_off": cut_offs
    }
    with open(f'{tuning_folder}/ranges.json', 'w') as f:
        json.dump(ranges, f)

    combinations = len(N_g_values) * len(intial_temperatures) * len(cooling_rates) * len(cut_offs)

    output = []

    try:
        for index, (N_g, initial_temperature, cooling_rate, cut_off) in tqdm(enumerate(itertools.product(N_g_values, intial_temperatures, cooling_rates, cut_offs)), total=combinations):
            logging.info(f'Running parameter selection for N_g={N_g}, initial_temperature={initial_temperature}, cooling_rate={cooling_rate}, cut_off={cut_off}')

            C_g = compute_group_modes(corr_eigenvalues, corr_eigenvectors, N_g)
            best_order, best_energy, iteration_count = simulated_annealing_ordering(
                C_g,
                cutoff=cut_off,
                initial_temp=initial_temperature,
                cooling_rate=cooling_rate,
                iterations=200000,
                tol=10,
                patience=1000,
                return_history=False,
            )

            output.append({
                'index': index,
                'best_energy': best_energy,
                'iteration': iteration_count,
                'N_g': N_g,
                'initial_temperature': initial_temperature,
                'cooling_rate': cooling_rate,
                'cut_off': cut_off,
                'best_order': best_order,
            })

    except Exception as e:
        logging.error(f'Error: {e}')

    logging.info(f'Saving output to JSON')
    with open(f'{tuning_folder}/parameter_selection_output.json', 'w') as f:
        json.dump(output, f)
