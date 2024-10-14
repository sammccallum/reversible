import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": 18,
        "text.latex.preamble": r"\usepackage{lmodern, amsmath, amssymb, amsfonts}",
        "legend.fontsize": 12,
    }
)


def eig_euler(l, h, a):
    """
    Reversible Euler eigenvalues obtained by calculate_eigenvalues function.
    """
    eig1 = (
        a**2 * h**2 / 2
        - a * h * l / 2
        + a * h / 2
        + l / 2
        - np.sqrt(
            a**4 * h**4
            - 2 * a**3 * h**3 * l
            + 2 * a**3 * h**3
            + a**2 * h**2 * l**2
            + 3 * a**2 * h**2
            - 2 * a * h * l**2
            + 2 * a * h
            + l**2
            - 2 * l
            + 1
        )
        / 2
        + 1 / 2
    )
    eig2 = (
        a**2 * h**2 / 2
        - a * h * l / 2
        + a * h / 2
        + l / 2
        + np.sqrt(
            a**4 * h**4
            - 2 * a**3 * h**3 * l
            + 2 * a**3 * h**3
            + a**2 * h**2 * l**2
            + 3 * a**2 * h**2
            - 2 * a * h * l**2
            + 2 * a * h
            + l**2
            - 2 * l
            + 1
        )
        / 2
        + 1 / 2
    )
    return eig1, eig2


def eig_midpoint(l, h, a):
    """
    Reversible Midpoint eigenvalues obtained by calculate_eigenvalues function.
    """
    eig1 = (
        -(a**4) * h**4 / 8
        - a**2 * h**2 * l / 4
        + 3 * a**2 * h**2 / 4
        - a * h * l / 2
        + a * h / 2
        + l / 2
        - np.sqrt(
            a**8 * h**8
            + 4 * a**6 * h**6 * l
            - 12 * a**6 * h**6
            + 8 * a**5 * h**5 * l
            - 8 * a**5 * h**5
            + 4 * a**4 * h**4 * l**2
            - 32 * a**4 * h**4 * l
            + 28 * a**4 * h**4
            + 16 * a**3 * h**3 * l**2
            - 64 * a**3 * h**3 * l
            + 48 * a**3 * h**3
            + 64 * a**2 * h**2 * l**2
            - 64 * a**2 * h**2 * l
            + 64 * a**2 * h**2
            - 32 * a * h * l**2
            + 32 * a * h
            + 16 * l**2
            - 32 * l
            + 16
        )
        / 8
        + 1 / 2
    )
    eig2 = (
        -(a**4) * h**4 / 8
        - a**2 * h**2 * l / 4
        + 3 * a**2 * h**2 / 4
        - a * h * l / 2
        + a * h / 2
        + l / 2
        + np.sqrt(
            a**8 * h**8
            + 4 * a**6 * h**6 * l
            - 12 * a**6 * h**6
            + 8 * a**5 * h**5 * l
            - 8 * a**5 * h**5
            + 4 * a**4 * h**4 * l**2
            - 32 * a**4 * h**4 * l
            + 28 * a**4 * h**4
            + 16 * a**3 * h**3 * l**2
            - 64 * a**3 * h**3 * l
            + 48 * a**3 * h**3
            + 64 * a**2 * h**2 * l**2
            - 64 * a**2 * h**2 * l
            + 64 * a**2 * h**2
            - 32 * a * h * l**2
            + 32 * a * h
            + 16 * l**2
            - 32 * l
            + 16
        )
        / 8
        + 1 / 2
    )
    return eig1, eig2


def eig_rk3(l, h, a):
    """
    Reversible RK3 eigenvalues obtained by calculate_eigenvalues function.
    """
    eig1 = (
        a**6 * h**6 / 72
        + a**4 * h**4 / 24
        - a**3 * h**3 * l / 12
        + a**3 * h**3 / 12
        - a**2 * h**2 * l / 4
        + 3 * a**2 * h**2 / 4
        - a * h * l / 2
        + a * h / 2
        + l / 2
        - np.sqrt(
            a**12 * h**12
            + 6 * a**10 * h**10
            - 12 * a**9 * h**9 * l
            + 12 * a**9 * h**9
            - 36 * a**8 * h**8 * l
            + 117 * a**8 * h**8
            - 108 * a**7 * h**7 * l
            + 108 * a**7 * h**7
            + 36 * a**6 * h**6 * l**2
            - 108 * a**6 * h**6 * l
            + 432 * a**6 * h**6
            + 216 * a**5 * h**5 * l**2
            - 1080 * a**5 * h**5 * l
            + 864 * a**5 * h**5
            + 756 * a**4 * h**4 * l**2
            - 2592 * a**4 * h**4 * l
            + 3564 * a**4 * h**4
            + 864 * a**3 * h**3 * l**2
            - 5184 * a**3 * h**3 * l
            + 4320 * a**3 * h**3
            + 5184 * a**2 * h**2 * l**2
            - 5184 * a**2 * h**2 * l
            + 5184 * a**2 * h**2
            - 2592 * a * h * l**2
            + 2592 * a * h
            + 1296 * l**2
            - 2592 * l
            + 1296
        )
        / 72
        + 1 / 2
    )
    eig2 = (
        a**6 * h**6 / 72
        + a**4 * h**4 / 24
        - a**3 * h**3 * l / 12
        + a**3 * h**3 / 12
        - a**2 * h**2 * l / 4
        + 3 * a**2 * h**2 / 4
        - a * h * l / 2
        + a * h / 2
        + l / 2
        + np.sqrt(
            a**12 * h**12
            + 6 * a**10 * h**10
            - 12 * a**9 * h**9 * l
            + 12 * a**9 * h**9
            - 36 * a**8 * h**8 * l
            + 117 * a**8 * h**8
            - 108 * a**7 * h**7 * l
            + 108 * a**7 * h**7
            + 36 * a**6 * h**6 * l**2
            - 108 * a**6 * h**6 * l
            + 432 * a**6 * h**6
            + 216 * a**5 * h**5 * l**2
            - 1080 * a**5 * h**5 * l
            + 864 * a**5 * h**5
            + 756 * a**4 * h**4 * l**2
            - 2592 * a**4 * h**4 * l
            + 3564 * a**4 * h**4
            + 864 * a**3 * h**3 * l**2
            - 5184 * a**3 * h**3 * l
            + 4320 * a**3 * h**3
            + 5184 * a**2 * h**2 * l**2
            - 5184 * a**2 * h**2 * l
            + 5184 * a**2 * h**2
            - 2592 * a * h * l**2
            + 2592 * a * h
            + 1296 * l**2
            - 2592 * l
            + 1296
        )
        / 72
        + 1 / 2
    )
    return eig1, eig2


def eig_rk4(l, h, a):
    """
    Reversible RK4 eigenvalues obtained by calculate_eigenvalues function.
    """
    eig1 = (
        -(a**8) * h**8 / 1152
        - a**6 * h**6 / 144
        - a**4 * h**4 * l / 48
        + a**4 * h**4 / 16
        - a**3 * h**3 * l / 12
        + a**3 * h**3 / 12
        - a**2 * h**2 * l / 4
        + 3 * a**2 * h**2 / 4
        - a * h * l / 2
        + a * h / 2
        + l / 2
        - np.sqrt(
            a**16 * h**16
            + 16 * a**14 * h**14
            + 48 * a**12 * h**12 * l
            - 80 * a**12 * h**12
            + 192 * a**11 * h**11 * l
            - 192 * a**11 * h**11
            + 960 * a**10 * h**10 * l
            - 2880 * a**10 * h**10
            + 2688 * a**9 * h**9 * l
            - 2688 * a**9 * h**9
            + 576 * a**8 * h**8 * l**2
            - 9792 * a**8 * h**8
            + 4608 * a**7 * h**7 * l**2
            - 9216 * a**7 * h**7 * l
            + 4608 * a**7 * h**7
            + 23040 * a**6 * h**6 * l**2
            - 110592 * a**6 * h**6 * l
            + 124416 * a**6 * h**6
            + 82944 * a**5 * h**5 * l**2
            - 331776 * a**5 * h**5 * l
            + 248832 * a**5 * h**5
            + 276480 * a**4 * h**4 * l**2
            - 774144 * a**4 * h**4 * l
            + 940032 * a**4 * h**4
            + 221184 * a**3 * h**3 * l**2
            - 1327104 * a**3 * h**3 * l
            + 1105920 * a**3 * h**3
            + 1327104 * a**2 * h**2 * l**2
            - 1327104 * a**2 * h**2 * l
            + 1327104 * a**2 * h**2
            - 663552 * a * h * l**2
            + 663552 * a * h
            + 331776 * l**2
            - 663552 * l
            + 331776
        )
        / 1152
        + 1 / 2
    )
    eig2 = (
        -(a**8) * h**8 / 1152
        - a**6 * h**6 / 144
        - a**4 * h**4 * l / 48
        + a**4 * h**4 / 16
        - a**3 * h**3 * l / 12
        + a**3 * h**3 / 12
        - a**2 * h**2 * l / 4
        + 3 * a**2 * h**2 / 4
        - a * h * l / 2
        + a * h / 2
        + l / 2
        + np.sqrt(
            a**16 * h**16
            + 16 * a**14 * h**14
            + 48 * a**12 * h**12 * l
            - 80 * a**12 * h**12
            + 192 * a**11 * h**11 * l
            - 192 * a**11 * h**11
            + 960 * a**10 * h**10 * l
            - 2880 * a**10 * h**10
            + 2688 * a**9 * h**9 * l
            - 2688 * a**9 * h**9
            + 576 * a**8 * h**8 * l**2
            - 9792 * a**8 * h**8
            + 4608 * a**7 * h**7 * l**2
            - 9216 * a**7 * h**7 * l
            + 4608 * a**7 * h**7
            + 23040 * a**6 * h**6 * l**2
            - 110592 * a**6 * h**6 * l
            + 124416 * a**6 * h**6
            + 82944 * a**5 * h**5 * l**2
            - 331776 * a**5 * h**5 * l
            + 248832 * a**5 * h**5
            + 276480 * a**4 * h**4 * l**2
            - 774144 * a**4 * h**4 * l
            + 940032 * a**4 * h**4
            + 221184 * a**3 * h**3 * l**2
            - 1327104 * a**3 * h**3 * l
            + 1105920 * a**3 * h**3
            + 1327104 * a**2 * h**2 * l**2
            - 1327104 * a**2 * h**2 * l
            + 1327104 * a**2 * h**2
            - 663552 * a * h * l**2
            + 663552 * a * h
            + 331776 * l**2
            - 663552 * l
            + 331776
        )
        / 1152
        + 1 / 2
    )
    return eig1, eig2


def coupling_matrix(R):
    """
    Reversible coupling matrix (SymPy).
    """
    l, h, a = sp.symbols("l h a")
    M = sp.Matrix(
        [
            [l, 1 - l + R(h * a)],
            [-l * R(-h * a), 1 + (1 - l) * R(h * a) - R(-h * a) * R(h * a)],
        ]
    )
    return M


def calculate_eigenvalues(M):
    """
    Calculate Reversible {solver} eigenvalues from (SymPy) coupling matrix M.
    """
    return M.eigenvals()


def euler_stability(h, a):
    return 1 + h * a


def midpoint_stability(h, a):
    return 1 + h * a + ((h * a) ** 2) / 2


def rk3_stability(h, a):
    return 1 + h * a + ((h * a) ** 2) / 2 + ((h * a) ** 3) / 6


def rk4_stability(h, a):
    return 1 + h * a + ((h * a) ** 2) / 2 + ((h * a) ** 3) / 6 + ((h * a) ** 4) / 24


if __name__ == "__main__":
    l = 0.8
    h = 1

    x = np.linspace(-3 / h, 0, 400)
    y = np.linspace(-3 / h, 3 / h, 400)
    X, Y = np.meshgrid(x, y)
    a = X + 1j * Y

    # Reversible
    euler1, euler2 = np.abs(eig_euler(l, h, a))
    midpoint1, midpoint2 = np.abs(eig_midpoint(l, h, a))
    rk3_1, rk3_2 = np.abs(eig_rk3(l, h, a))
    rk4_1, rk4_2 = np.abs(eig_rk4(l, h, a))

    # Standard
    mag_euler = np.abs(euler_stability(h, a))
    mag_midpoint = np.abs(midpoint_stability(h, a))
    mag_rk3 = np.abs(rk3_stability(h, a))
    mag_rk4 = np.abs(rk4_stability(h, a))

    fig, axs = plt.subplots(1, 4, figsize=(12, 3.5), sharey=True)

    # Euler
    axs[0].contour(
        X, Y, np.maximum(euler1, euler2), levels=[1], colors="tab:red", label="Euler"
    )
    axs[0].contour(X, Y, mag_euler, levels=[1], colors="tab:blue", label="Euler")
    axs[0].text(
        0.5,
        1.1,
        r"Euler",
        transform=axs[0].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="center",
    )
    axs[0].set_xlabel(r"Re($\alpha$)")
    axs[0].set_ylabel(r"Im($\alpha$)")
    legend_handles = [
        Line2D([0], [0], color="tab:blue", linestyle="solid", label=r"Original"),
        Line2D(
            [0],
            [0],
            color="tab:red",
            linestyle="solid",
            label=r"Reversible ($\lambda=0.8$)",
        ),
    ]
    axs[0].legend(handles=legend_handles, loc="upper left", handlelength=1)

    # Midpoint
    axs[1].contour(
        X,
        Y,
        np.maximum(midpoint1, midpoint2),
        levels=[1],
        colors="tab:red",
        label="Midpoint",
    )
    axs[1].contour(X, Y, mag_midpoint, levels=[1], colors="tab:blue", label="Midpoint")
    axs[1].text(
        0.5,
        1.1,
        r"Midpoint",
        transform=axs[1].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="center",
    )
    axs[1].set_xlabel(r"Re($\alpha$)")

    # RK3
    axs[2].contour(
        X, Y, np.maximum(rk3_1, rk3_2), levels=[1], colors="tab:red", label="RK3"
    )
    axs[2].contour(X, Y, mag_rk3, levels=[1], colors="tab:blue", label="RK3")
    axs[2].text(
        0.5,
        1.1,
        r"RK3",
        transform=axs[2].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="center",
    )
    axs[2].set_xlabel(r"Re($\alpha$)")

    # RK4
    axs[3].contour(
        X, Y, np.maximum(rk4_1, rk4_2), levels=[1], colors="tab:red", label="RK4"
    )
    axs[3].contour(X, Y, mag_rk4, levels=[1], colors="tab:blue", label="RK4")
    axs[3].text(
        0.5,
        1.1,
        r"RK4",
        transform=axs[3].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="center",
    )
    axs[3].set_xlabel(r"Re($\alpha$)")

    plt.tight_layout()
    plt.savefig("stability.png", dpi=1000)
