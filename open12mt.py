import numpy as np
import matplotlib.pyplot as plt
import h5py


def plot_1d_from_npz(path):
    d = np.load(path, allow_pickle=True)
    x = d["x"]
    y = d["y"]

    label = str(d["label"]) if "label" in d else None
    color = str(d["color"]) if "color" in d else None
    linestyle = str(d["linestyle"]) if "linestyle" in d else "-"
    linewidth = float(d["linewidth"]) if "linewidth" in d else 1.5
    alpha = float(d["alpha"]) if "alpha" in d else 1.0
    xlabel = str(d["xlabel"]) if "xlabel" in d else "x"
    ylabel = str(d["ylabel"]) if "ylabel" in d else "y"
    title = str(d["title"]) if "title" in d else "1D plot"

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    ax.plot(
        x, y,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if label and label != "_nolegend_":
        ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_1d_from_h5(path):
    with h5py.File(path, "r") as h5:
        x = h5["x"][:]
        y = h5["y"][:]
        label = h5.attrs.get("label", None)
        color = h5.attrs.get("color", None)
        linestyle = h5.attrs.get("linestyle", "-")
        linewidth = float(h5.attrs.get("linewidth", 1.5))
        alpha = float(h5.attrs.get("alpha", 1.0))
        xlabel = h5.attrs.get("xlabel", "x")
        ylabel = h5.attrs.get("ylabel", "y")
        title = h5.attrs.get("title", "1D plot")

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    ax.plot(
        x, y,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if label and label != "_nolegend_":
        ax.legend()
    plt.tight_layout()
    return fig, ax


# Użycie:
# fig, ax = plot_1d_from_npz("overlay_1d.npz")
# fig, ax = plot_1d_from_h5("overlay_1d.h5")
# plt.show()
