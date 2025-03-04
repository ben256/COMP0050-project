import os
from glob import glob

import numpy as np


def compute_log_returns(price_data, as_dataframe=False):
    returns = np.log(price_data / price_data.shift(1))
    returns.dropna(inplace=True)
    if as_dataframe:
        return returns
    else:
        return returns.values


def normalise(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))


def find_best_energies(all_data, values, name):
    best_energies = []

    for value in values:
        filtered_data = [x for x in all_data if x[name] == value]
        best_energy = filtered_data[np.argmin([x['best_energy'] for x in filtered_data])]['best_energy']
        best_energies.append(best_energy)

    return best_energies


def create_output_folder(output_dir, folder_name='tuning'):
    tuning_folders = glob(f'{output_dir}/{folder_name}_*')
    folder_num = [int(x.split('_')[-1]) for x in tuning_folders]
    if len(folder_num) > 0:
        count = max(folder_num) + 1
    else:
        count = 0
    folder_path = f'{output_dir}/{folder_name}_{count}/'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return folder_path

    else:
        raise FileExistsError
