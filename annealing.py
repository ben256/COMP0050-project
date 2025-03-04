import logging

import numpy as np
from numba import njit


def energy(order, C, cutoff):
    """Compute the total energy for a given ordering and correlation matrix C.

    order : list or array of indices representing the order of stocks.
    C     : correlation matrix (assumed symmetric).
    cutoff: cutoff value c_c to ignore small correlations.
    """
    n = len(order)
    total = 0.0
    # Loop over pairs (i, j) with i<j in the ordering
    for idx_i in range(n):
        i = order[idx_i]
        for idx_j in range(idx_i+1, n):
            j = order[idx_j]
            if C[i, j] > cutoff:
                total += C[i, j] * (idx_j - idx_i)
    return total


def vectorised_energy(order, C, cutoff):
    order = np.array(order)
    n = len(order)
    # Get indices for all pairs (i, j) with i < j
    i_idx, j_idx = np.triu_indices(n, k=1)
    # Map ordering indices to actual stock indices
    stocks_i = order[i_idx]
    stocks_j = order[j_idx]
    # Compute the differences in positions
    diff = j_idx - i_idx
    # Get the corresponding correlation values
    c_vals = C[stocks_i, stocks_j]
    # Only sum those above the cutoff
    mask = c_vals > cutoff
    return np.sum(c_vals[mask] * diff[mask])


@njit
def energy_numba(order, C, cutoff):
    n = len(order)
    total = 0.0
    for idx_i in range(n):
        i = order[idx_i]
        for idx_j in range(idx_i+1, n):
            j = order[idx_j]
            if C[i, j] > cutoff:
                total += C[i, j] * (idx_j - idx_i)
    return total


def simulated_annealing_ordering(C, cutoff=0.1, initial_temp=1.0, cooling_rate=0.9999, iterations=500000, tol=5, patience=1000, return_history=False):
    """
    Perform simulated annealing to optimise the ordering for block-diagonality.

    C            : correlation matrix (for example, the group matrix C_g).
    cutoff       : cutoff value to consider correlations significant.
    initial_temp : starting temperature for annealing.
    cooling_rate : factor by which to multiply the temperature each iteration.
    iterations   : total number of iterations to run.

    Returns:
    best_order   : the optimised ordering (a list of indices).
    best_energy  : energy value corresponding to best_order.
    energy_history: list of energy values (optional, for monitoring).
    it: number of iterations.
    """

    n = C.shape[0]

    # Start with an initial ordering (0, 1, 2, ..., n-1)
    current_order = list(range(n))
    current_energy = energy_numba(current_order, C, cutoff)
    best_order = current_order.copy()
    best_energy = current_energy
    temp = initial_temp
    energy_history = [current_energy]

    # Count iterations with negligible change.
    no_change_count = 0

    for it in range(iterations):
        # Propose a new ordering by removing one element and inserting it elsewhere.
        new_order = current_order.copy()
        i = np.random.randint(0, n)
        elem = new_order.pop(i)
        j = np.random.randint(0, n-1)
        new_order.insert(j, elem)

        new_energy = energy_numba(new_order, C, cutoff)
        delta_E = new_energy - current_energy

        # Accept new ordering if energy decreases, or with probability exp(-delta_E/temp)
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / temp):
            current_order = new_order
            current_energy = new_energy
            # Record if we found a new best
            if current_energy < best_energy:
                best_energy = current_energy
                best_order = current_order.copy()

        energy_history.append(current_energy)
        # Cool down the temperature
        temp *= cooling_rate

        if abs(energy_history[-1] - energy_history[-2]) < tol:
            no_change_count += 1
        else:
            no_change_count = 0  # Reset if a significant change occurs.

        if no_change_count >= patience:
            logging.info(f"Converged at iteration {it}")
            break

    if return_history:
        return best_order, best_energy, energy_history, it or 0

    return best_order, best_energy, it or 0
