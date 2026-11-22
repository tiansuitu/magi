# Nonlinear ODE coefficient approximation

The target is $B^\ast(t)=x^\ast(t)^2+t$ on $[0,1]$, where

$$
\dot{x}^\ast(t)=0.8\bigl(x^\ast(t)^2+t\bigr), \qquad x^\ast(0)=0.4.
$$

SciPy's DOP853 solver supplies the true trajectory for data generation and error evaluation, using dense output with relative tolerance $10^{-12}$ and absolute tolerance $10^{-14}$. At $t_i=(i-1)/(n-1)$, both methods use the same observations $Y_i=x^\ast(t_i)+\sigma\varepsilon_i$, with independent standard normal errors $\varepsilon_i$.

## Estimators

- **Method A (smooth first)** fits a local line $a+b(t_i-t)$ to the observations at each evaluation time $t$, with Gaussian weights $\exp[-(t_i-t)^2/(2h_n^2)]$ and bandwidth $h_n=0.48n^{-1/5}$. Its fitted intercept gives $\widehat B_n^A(t)=\widehat a_n(t)^2+t$.
- **Method B (direct substitution)** computes $Y_i^2+t_i$ at each observation time and linearly interpolates those values. It uses the raw squared observations without smoothing or bias correction.

## Error and convergence rates

For each method $M$ and replication, the error is

$$
E_n^M=\left(\int_0^1 \bigl|\widehat B_n^M(t)-B^\ast(t)\bigr|^2\,dt\right)^{1/2}.
$$

Composite Simpson quadrature uses the endpoints and midpoint of every observation interval. The simulation averages the resulting norms over 300 replications for each combination of $n\in\{50,100,200,400,800,1600\}$ and $\sigma\in\{0.2,0.4\}$, using seed 2026.

For each method and noise level, the empirical exponent $p$ is the negative slope from regressing log mean error on $\log n$ across all six sizes. Each figure anchors the $n^{-2/5}$ benchmark to Method A's mean error at $n=50$. Method B's exponent describes its finite-sample trend. At fixed positive $\sigma$, squaring the observations introduces a mean bias of $\sigma^2$ at every node.

## Reproduce

With Python, NumPy, SciPy, Pandas, and Matplotlib installed, run

~~~bash
cd "C:\Users\Tiansui Tu\Documents\GitHub\magi\ode-coefficient-simulation"
python simulation.py
~~~

The script writes `results/L2_error_sigma_0.2.png` and `results/L2_error_sigma_0.4.png`. The table `results/L2_error_summary.csv` contains one row per noise level and sample size, with both mean errors and both fitted exponents. Each noise level's exponents repeat across its six rows.
