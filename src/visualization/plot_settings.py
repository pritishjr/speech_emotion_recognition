import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

def global_plot_settings():
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.format": "png"
    })
