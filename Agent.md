# Agent Session — Monte Carlo π

## Model
- `inkling-bf16`

## Input Prompt
Write `pi_mc.py`: Monte Carlo π estimation with `numpy`, `matplotlib`, publication-quality figure. Vectorized, no Python loops. Args: `--samples` (default 1_000_000), `--seed` (default 0), `--out`. Compute `pi_hat = 4p`, SE, fail if `|pi_hat - π| / SE > 3`. Figure (13×6, dpi 150, constrained layout): left scatter (first 5,000), top-right running π (log x, ±2σ band, true π dashed), bottom-right `|π̂ - π|` vs N (log-log, reference `σ/√N`). No `plt.show()`.

## Key Constraints
- Vectorized `numpy` (`np.random.default_rng`), no loops over samples
- Headless: `plt.savefig`, never `plt.show()`
- Fail (`SystemExit`) if z-score > 3
- ~2,000 log-spaced points for running plot (`np.cumsum` then logspace sampling)
- Clean style: no top/right spines, light grid, panel titles + axis labels, suptitle with final estimate ± error

## Commands Run & Outputs

### Default
```
python pi_mc.py
```
Output:
```
N = 1000000
pi_hat = 3.141368 +/- 0.001642
Standard errors from pi: 0.137
Figure saved to pi_monte_carlo.png
```

### Small sample
```
python pi_mc.py -n 1000 --seed 3
```
Output:
```
N = 1000
pi_hat = 3.188000 +/- 0.050879
Standard errors from pi: 0.912
Figure saved to pi_monte_carlo.png
```
