import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": 18,
        "text.latex.preamble": r"\usepackage{lmodern, amsmath, amssymb, amsfonts}",
        "legend.fontsize": 16,
    }
)

if __name__ == "__main__":
    otd_error = np.loadtxt("../results/otd_error.txt", delimiter=",", skiprows=1)
    rev_error = np.loadtxt("../results/rev_error.txt", delimiter=",", skiprows=1)

    dto_mem = np.loadtxt("../results/dto_mem_usage.txt", delimiter=",", skiprows=1)
    rev_mem = np.loadtxt("../results/rev_mem_usage.txt", delimiter=",", skiprows=1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))

    ax[0].plot(
        otd_error[:, 0],
        np.mean(otd_error[:, 1:], axis=1) * 1000,
        marker=".",
        linestyle="-",
        color="tab:red",
        label="Optimise-then-Discretise",
    )
    ax[0].plot(
        rev_error[:, 0],
        np.mean(rev_error[:, 1:], axis=1) * 1000,
        marker=".",
        linestyle="-",
        color="tab:blue",
        label="Reversible Method",
    )
    ax[0].legend()
    ax[0].set_xlabel(r"Integration Time, $T$")
    ax[0].set_ylabel(r"$L^1$ Gradient Error ($10^{-3}$)")
    ax[0].text(
        0.5,
        -0.18,
        "(a)",
        transform=ax[0].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="center",
    )

    ax[1].plot(
        dto_mem[:, 0],
        np.mean(dto_mem[:, 1:] / 1024, axis=1),
        linestyle="-",
        marker=".",
        color="tab:red",
        label=r"Discretise-then-Optimise",
    )

    ax[1].plot(
        rev_mem[:, 0],
        np.mean(rev_mem[:, 1:] / 1024, axis=1),
        linestyle="-",
        marker=".",
        color="tab:blue",
        label=r"Reversible Method",
    )
    ax[1].set_xlabel(r"Integration time, $T$")
    ax[1].set_ylabel(r"Memory Usage (GB)")
    ax[1].legend()
    ax[1].text(
        0.5,
        -0.18,
        "(b)",
        transform=ax[1].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="center",
    )

    plt.tight_layout()
    plt.savefig("error_and_memory.png", dpi=1000)
