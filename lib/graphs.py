import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lib.utils import normalise, find_best_energies


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


def plot_heat_map(C_g, best_order, stock_mapping, sector_mapping, output_dir):
    Cg_sorted = C_g[np.ix_(best_order, best_order)]

    # Create a mapping from new positions to sectors
    position_to_sector = {}
    for pos, orig_idx in enumerate(best_order):
        position_to_sector[pos] = sector_mapping[orig_idx]

    # Group positions by sector
    sector_positions = {}
    for pos, sector in position_to_sector.items():
        if sector not in sector_positions:
            sector_positions[sector] = []
        sector_positions[sector].append(pos)

    # Calculate median position for each sector
    sector_middles = {sector: np.median(positions) for sector, positions in sector_positions.items()}

    fig, ax = plt.subplots(figsize=(8, 7), tight_layout=True)
    im = ax.imshow(Cg_sorted, interpolation='nearest', aspect='auto', vmin=0.05, vmax=0.25)

    for sector, middle in sector_middles.items():
        ax.scatter(middle, middle, color='tab:red', marker='x', s=100)
        ax.text(middle + 10, middle - 10, sector, fontsize=8, color='white',
                bbox=dict(facecolor='tab:red', alpha=0.7, edgecolor='none', pad=1))

    ax.set_title("Optimized Group Correlation Matrix")
    ax.set_xlabel("Stock index (ordered)")
    ax.set_ylabel("Stock index (ordered)")

    plt.colorbar(im, label='Correlation')
    # plt.savefig(f'{output_dir}/heat_map.png', dpi=300)
    plt.show()
    plt.close()


def plot_heat_map_with_kde(C_g, best_order, stock_mapping, sector_mapping, output_dir):
    Cg_sorted = C_g[np.ix_(best_order, best_order)]

    # Create a mapping from new positions to sectors
    position_to_sector = {}
    for pos, orig_idx in enumerate(best_order):
        position_to_sector[pos] = sector_mapping[orig_idx]

    # Group positions by sector
    sector_positions = {}
    for pos, sector in position_to_sector.items():
        if sector not in sector_positions:
            sector_positions[sector] = []
        sector_positions[sector].append(pos)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8), tight_layout=True, width_ratios=[3, 1])
    im = ax[0].imshow(Cg_sorted, interpolation='nearest', aspect='auto', vmin=0.05, vmax=0.25, origin='lower')

    for sector, positions in sector_positions.items():
        middle = np.median(positions)
        sns.kdeplot(ax=ax[1], y=positions, fill=True, alpha=0.3, label=sector)
        ax[0].scatter(middle, 0, color='tab:red', marker='x', s=100)
        ax[1].text(0.005, middle, sector, fontsize=8, color='white',
                   bbox=dict(facecolor='tab:red', alpha=0.7, edgecolor='none', pad=1))

    ax[1].set_ylim(0, len(Cg_sorted))
    ax[0].set_title("Optimized Group Correlation Matrix")
    ax[0].set_xlabel("Stock index (ordered)")
    ax[0].set_ylabel("Stock index (ordered)")

    plt.colorbar(im, ax=ax[0], label='Correlation')
    # plt.savefig(f'{output_dir}/heat_map_with_kde.png', dpi=300)
    plt.show()
    plt.close()


def plot_heat_map_with_boxplot(C_g, best_order, stock_mapping, sector_mapping, output_dir):
    Cg_sorted = C_g[np.ix_(best_order, best_order)]

    # Create a mapping from new positions to sectors
    position_to_sector = {}
    for pos, orig_idx in enumerate(best_order):
        position_to_sector[pos] = sector_mapping[orig_idx]

    # Group positions by sector
    sector_positions = {}
    for pos, sector in position_to_sector.items():
        if sector not in sector_positions:
            sector_positions[sector] = []
        sector_positions[sector].append(pos)

    fig, ax = plt.subplots(1, 2, figsize=(15, 8), tight_layout=True)
    im = ax[0].imshow(Cg_sorted, interpolation='nearest', aspect='auto', vmin=0.05, vmax=0.25, origin='lower')

    box_plot_data = sector_positions

    for sector, positions in sector_positions.items():
        middle = np.median(positions)
        ax[0].axhline(middle, color='tab:orange', linestyle='--', linewidth=0.5)
        ax[1].axhline(middle, color='tab:orange', linestyle='--', linewidth=0.5)

    ax[1].boxplot(box_plot_data.values(), labels=box_plot_data.keys())
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha='right')

    ax[1].set_ylim(0, len(Cg_sorted))
    ax[0].set_title("Optimized Group Correlation Matrix")
    ax[0].set_xlabel("Stock index (ordered)")
    ax[0].set_ylabel("Stock index (ordered)")

    plt.colorbar(im, ax=ax[0], label='Correlation')
    # plt.savefig(f'{output_dir}/heat_map_with_boxplot.png', dpi=300)
    plt.show()
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
