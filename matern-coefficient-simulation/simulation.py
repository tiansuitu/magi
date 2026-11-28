"""Compare the original A/B estimators with Method C: Matérn GP regression."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson
from scipy.linalg import cho_factor, cho_solve

SAMPLE_SIZES = (50, 100, 200, 400, 800, 1600)
NOISE_LEVELS = (0.2, 0.4)
REPLICATIONS = 300
SEED = 2026
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def matern(s, t):
    r = np.abs(s[:, None] - t[None, :])
    return (1 + r + r**2 / 3) * np.exp(-r)


def local_linear_weights(obs_t, grid, bandwidth):
    u = (obs_t[None, :] - grid[:, None]) / bandwidth
    k = np.exp(-0.5 * u**2)
    s0 = k.sum(axis=1, keepdims=True)
    s1 = (k * u).sum(axis=1, keepdims=True)
    s2 = (k * u**2).sum(axis=1, keepdims=True)
    return k * (s2 - s1 * u) / (s0 * s2 - s1**2)


def l2_errors(estimate, truth, grid):
    return np.sqrt(simpson((estimate - truth[:, None])**2, x=grid, axis=0))


def simulate():
    rng = np.random.default_rng(SEED)
    sol = solve_ivp(lambda t, x: 0.8 * (x**2 + t), (0., 1.), [0.4],
                    method="DOP853", dense_output=True, rtol=1e-12, atol=1e-14)
    if not sol.success:
        raise RuntimeError(sol.message)
    rows = []
    for n in SAMPLE_SIZES:
        grid = np.linspace(0., 1., 2*n-1)
        obs_t = grid[::2]
        x_true = sol.sol(grid)[0]
        b_true = x_true**2 + grid
        weights = local_linear_weights(obs_t, grid, 0.48*n**(-1/5))
        C = matern(obs_t, obs_t)
        K_eval = matern(grid, obs_t)
        for sigma in NOISE_LEVELS:
            # Same RNG call order, grid and DGP as the original simulation.
            y = x_true[::2, None] + sigma*rng.standard_normal((n, REPLICATIONS))
            a = (weights @ y)**2 + grid[:, None]
            nodal = y**2 + obs_t[:, None]
            b = np.empty_like(a)
            b[::2] = nodal
            b[1::2] = 0.5*(nodal[:-1] + nodal[1:])
            # One Cholesky factorization per (n, sigma), all 300 RHS at once.
            factor = cho_factor(C + sigma**2*np.eye(n))
            alpha = cho_solve(factor, y)
            c = (K_eval @ alpha)**2 + grid[:, None]
            row = {"sigma": sigma, "n": n}
            for name, estimate in zip("ABC", (a, b, c)):
                errors = l2_errors(estimate, b_true, grid)
                row[f"mean_l2_{name}"] = errors.mean()
                row[f"mcse_{name}"] = errors.std(ddof=1)/np.sqrt(REPLICATIONS)
            diff = l2_errors(c, b_true, grid) - l2_errors(a, b_true, grid)
            row["paired_C_minus_A_mcse"] = diff.std(ddof=1)/np.sqrt(REPLICATIONS)
            rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["sigma", "n"], ignore_index=True)
    for sigma, group in summary.groupby("sigma"):
        for name in "ABC":
            slope, _ = np.polyfit(np.log(group.n), np.log(group[f"mean_l2_{name}"]), 1)
            summary.loc[group.index, f"exponent_{name}"] = -slope
    return summary


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    summary = simulate()
    summary.to_csv(RESULTS_DIR / "L2_error_summary.csv", index=False, float_format="%.10g")
    for sigma, group in summary.groupby("sigma"):
        fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
        for name, label, marker in zip("ABC", ("A: Gaussian local linear", "B: direct interpolation", "C: Matérn GP"), ("o", "s", "^")):
            ax.loglog(group.n, group[f"mean_l2_{name}"], marker+"-", label=label)
        ax.loglog(group.n, group.mean_l2_A.iloc[0]*(group.n/group.n.iloc[0])**(-2/5), "k--", label=r"$n^{-2/5}$ reference")
        ax.set_xticks(group.n, labels=group.n.astype(str))
        ax.set(xlabel="Sample size n", ylabel="Monte Carlo mean L2 error", title=f"Noise level sigma = {sigma:.1f}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(RESULTS_DIR / f"L2_error_sigma_{sigma:.1f}.png", dpi=200)
        plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
