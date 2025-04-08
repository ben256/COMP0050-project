import itertools
import json

from tqdm import tqdm

from lib.annealing import simulated_annealing_ordering
from lib.correlation import *
from lib.data_processing import *
from lib.graphs import *
from lib.utils import *


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

    N_g_values = [19]
    initial_temperatures = [1.5]
    cooling_rates = [0.9998]
    cut_offs = [0.1]
    # N_g_values = [19]
    # intial_temperatures = [0.5]
    # cooling_rates = [0.9997]

    # cut_offs = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5]

    ranges = {
        "N_g": N_g_values,
        "initial_temperature": initial_temperatures,
        "cooling_rate": cooling_rates,
        "cut_off": cut_offs
    }
    with open(f'{tuning_folder}/ranges.json', 'w') as f:
        json.dump(ranges, f)

    combinations = len(N_g_values) * len(initial_temperatures) * len(cooling_rates) * len(cut_offs)

    output = []

    try:
        for index, (N_g, initial_temperature, cooling_rate, cut_off) in tqdm(enumerate(itertools.product(N_g_values, initial_temperatures, cooling_rates, cut_offs)), total=combinations):
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

    plot_comparison_graph(output, ranges, tuning_folder, exclude_cut_off=False)

    logging.info(f'Saving output to JSON')
    with open(f'{tuning_folder}/parameter_selection_output.json', 'w') as f:
        json.dump(output, f)
