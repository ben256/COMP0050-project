import numpy as np
from matplotlib import pyplot as plt

from utils import normalise, find_best_energies


def plot_energy_history(energy_history, output_dir):
    fig, ax = plt.subplots(figsize=(9,6), tight_layout=True)

    ax.plot(energy_history, label='Energy')

    ax.set_title('Energy History')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Energy')

    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor', linestyle='-', alpha=0.3)
    plt.minorticks_on()
    plt.legend()
    plt.savefig(f'{output_dir}/energy_history.png', dpi=300)
    plt.close()


def plot_heat_map(C_g, best_order, output_dir):
    Cg_sorted = C_g[np.ix_(best_order, best_order)]

    fig, ax = plt.subplots(figsize=(8, 7), tight_layout=True)
    im = ax.imshow(Cg_sorted, interpolation='nearest', aspect='auto', vmin=0.05, vmax=0.25)

    ax.set_title("Optimized Group Correlation Matrix")
    ax.set_xlabel("Stock index (ordered)")
    ax.set_ylabel("Stock index (ordered)")

    plt.colorbar(im, label='Correlation')
    plt.savefig(f'{output_dir}/heat_map.png', dpi=300)
    plt.close()


def plot_comparison_graph(output, ranges, output_dir, exclude_cut_off=True):

    if exclude_cut_off:
        ranges.pop('cut_off')

    fig, ax = plt.subplots(figsize=(9,6), tight_layout=True)

    for key, value in ranges.items():
        x = normalise(value)
        y = find_best_energies(output, value, key)
        ax.plot(x, y, label=key, marker='o')

        for x, y, value in zip(x, y, value):
            text = f'{value}'
            ax.annotate(text, (x, y), xytext=(3, 3), textcoords="offset points")

    ax.set_xticks([])
    ax.set_title(f'Parameter Comparison{"" if not exclude_cut_off else " (Cut Off Excluded)"}')
    ax.set_ylabel('Best Energy')
    ax.set_xlabel('Normalised Parameter Values')

    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor', linestyle='-', alpha=0.3)
    plt.minorticks_on()
    plt.legend()
    plt.savefig(f'{output_dir}/comparison_plot{"" if not exclude_cut_off else "_no_co"}.png', dpi=300)
    plt.close()
