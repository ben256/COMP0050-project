import logging

from group_correlation_test import run_group_correlation
from parameter_selection import run_parameter_selection
from sparse_pca import run_sparse_pca

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


if __name__ == '__main__':
    # run_sparse_pca()
    # run_group_correlation()
    run_parameter_selection()