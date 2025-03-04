import numpy as np


def compute_correlation_matrix(timeseries_data):
    return np.corrcoef(timeseries_data, rowvar=False)


def compute_covariance_matrix(timeseries_data):
    return np.cov(timeseries_data, rowvar=False)


def compute_eigenvalues(matrix, sort=True):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Eigen values sorted in descending order
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors


def find_bulk_eigenvalues(eigenvalues, Q):
    lambda_max = (1 + 1/np.sqrt(Q))**2
    lambda_min = (1 - 1/np.sqrt(Q))**2

    bulk_eigenvalues = eigenvalues[:-1]
    bulk_eigenvalues = bulk_eigenvalues[(bulk_eigenvalues >= lambda_min) & (bulk_eigenvalues <= lambda_max)]
    return bulk_eigenvalues


def compute_market_mode(eigenvalues, eigenvectors):
    return eigenvalues[0] * np.outer(eigenvectors[:, 0], eigenvectors[:, 0])


def compute_group_modes(eigenvalues, eigenvectors, N_g):
    return sum(eigenvalues[i] * np.outer(eigenvectors[:, i], eigenvectors[:, i]) for i in range(1, N_g))


def compute_residual_modes(correlation_matrix, market_mode, group_modes):
    return (correlation_matrix - market_mode - group_modes).to_numpy()
