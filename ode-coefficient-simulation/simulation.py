"""Compare two estimators of B(t) = x(t)^2 + t in a nonlinear ODE."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simpson, solve_ivp


SAMPLE_SIZES = (50, 100, 200, 400, 800, 1600)
NOISE_LEVELS = (0.2, 0.4)
REPLICATIONS = 300
SEED = 2026
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def solve_truth():
    """Return the dense ODE solution for data generation and error evaluation."""
    solution = solve_ivp(
        lambda t, x: 0.8 * (x**2 + t),
        (0.0, 1.0),
        [0.4],
        method="DOP853",
        dense_output=True,
        rtol=1e-12,
        atol=1e-14,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.sol


def local_linear_weights(obs_t, eval_t, bandwidth):
    """Build one row of local-linear intercept weights per evaluation time."""
    u = (obs_t[None, :] - eval_t[:, None]) / bandwidth
    kernel = np.exp(-0.5 * u**2)
    s0 = kernel.sum(axis=1, keepdims=True)
    s1 = (kernel * u).sum(axis=1, keepdims=True)
    s2 = (kernel * u**2).sum(axis=1, keepdims=True)

    # Solving the weighted normal equations in the basis (1, u) gives the
    # intercept as a weighted sum of observations, including at the boundaries.
    return kernel * (s2 - s1 * u) / (s0 * s2 - s1**2)


def mean_l2_error(estimate, truth, grid):
    """Integrate each replication's squared error, take its root, then average."""
    squared_error = (estimate - truth[:, None]) ** 2
    integrals = simpson(squared_error, x=grid, axis=0)
    return np.sqrt(integrals).mean()


def simulate():
    rng = np.random.default_rng(SEED)
    trajectory = solve_truth()
    rows = []

    for n in SAMPLE_SIZES:
        # Even indices are observation times and odd indices are their midpoints.
        grid = np.linspace(0.0, 1.0, 2 * n - 1)
        obs_t = grid[::2]
        x_true = trajectory(grid)[0]
        b_true = x_true**2 + grid
        weights = local_linear_weights(obs_t, grid, 0.48 * n ** (-1 / 5))

        for sigma in NOISE_LEVELS:
            # Each column is one replication, shared by both methods.
            noise = rng.standard_normal((n, REPLICATIONS))
            y = x_true[::2, None] + sigma * noise

            b_a = (weights @ y) ** 2 + grid[:, None]

            # Linear interpolation at a midpoint averages the two nodal B values.
            nodal_b = y**2 + obs_t[:, None]
            b_b = np.empty_like(b_a)
            b_b[::2] = nodal_b
            b_b[1::2] = 0.5 * (nodal_b[:-1] + nodal_b[1:])

            rows.append(
                {
                    "sigma": sigma,
                    "n": n,
                    "mean_l2_A": mean_l2_error(b_a, b_true, grid),
                    "mean_l2_B": mean_l2_error(b_b, b_true, grid),
                }
            )

    return pd.DataFrame(rows).sort_values(["sigma", "n"], ignore_index=True)


def plot_errors(group, sigma):
    n = group["n"].to_numpy()
    mean_a = group["mean_l2_A"].to_numpy()
    reference = mean_a[0] * (n / n[0]) ** (-2 / 5)

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.loglog(n, mean_a, "o-", color="#0072B2", label="Method A (smooth first)")
    ax.loglog(
        n, group["mean_l2_B"], "s-", color="#D55E00",
        label="Method B (direct substitution)",
    )
    ax.loglog(n, reference, "--", color="0.35", label=r"$n^{-2/5}$ reference")
    ax.set_xticks(n)
    ax.set_xticklabels(n)
    ax.set_xlabel(r"Sample size $n$")
    ax.set_ylabel(r"Monte Carlo mean $L^2$ error")
    ax.set_title(rf"Noise level $\sigma = {sigma:.1f}$")
    ax.grid(which="major", color="0.88", linewidth=0.6)
    ax.legend(frameon=False)
    fig.savefig(RESULTS_DIR / f"L2_error_sigma_{sigma:.1f}.png", dpi=300)
    plt.close(fig)


def main():
    summary = simulate()
    RESULTS_DIR.mkdir(exist_ok=True)
    for sigma, group in summary.groupby("sigma"):
        log_n = np.log(group["n"].to_numpy())
        for method in ("A", "B"):
            log_error = np.log(group[f"mean_l2_{method}"].to_numpy())
            slope, _ = np.polyfit(log_n, log_error, deg=1)
            # The exponent p uses the convention mean error proportional to n^(-p).
            summary.loc[group.index, f"exponent_{method}"] = -slope
        plot_errors(group, sigma)

    summary.to_csv(RESULTS_DIR / "L2_error_summary.csv", index=False)


if __name__ == "__main__":
    main()
