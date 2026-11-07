#!/usr/bin/env python3
"""Monte Carlo estimation of pi with publication-quality figure."""
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Estimate pi with Monte Carlo")
    parser.add_argument(
        "-n", "--samples", type=int, default=1_000_000, help="Number of samples"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed"
    )
    parser.add_argument(
        "--out", type=str, default="pi_monte_carlo.png", help="Output PNG path"
    )
    args = parser.parse_args()

    N = args.samples
    seed = args.seed
    out_path = args.out

    rng = np.random.default_rng(seed)

    # Vectorized uniform points in [0,1) x [0,1)
    x = rng.random(N)
    y = rng.random(N)

    inside = (x**2 + y**2) <= 1
    p = np.mean(inside)
    pi_hat = 4.0 * p
    se = 4.0 * math.sqrt(p * (1.0 - p) / N)

    # Distance from true pi in standard errors
    z = abs(pi_hat - math.pi) / se
    if z > 3:
        raise SystemExit(
            f"Estimate is more than 3 standard errors from pi: z={z:.3f}"
        )

    print(f"N = {N}")
    print(f"pi_hat = {pi_hat:.6f} +/- {se:.6f}")
    print(f"Standard errors from pi: {z:.3f}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), dpi=150, constrained_layout=True)
    ax_scatter = axes[0, 0]
    ax_running = axes[0, 1]
    ax_loglog = axes[1, 0]
    axes[1, 1].axis("off")  # hide extra panel; we use 3 active panels in a 2x2 layout
    # Actually user asked for: left panel, top right, bottom right.
    # Let's rearrange: left column spans both rows? No, 2x2 with hidden bottom-right is fine,
    # but let's instead use 2x2 and place bottom right in axes[1,1], top right in axes[0,1], left in axes[0,0].
    # That fits exactly: axes[0,0], axes[0,1], axes[1,1]. Hide axes[1,0] instead.
    axes[1, 0].axis("off")

    # Re-assign for clarity
    ax_left = axes[0, 0]
    ax_top_right = axes[0, 1]
    ax_bottom_right = axes[1, 1]

    # 1. Left panel: scatter of first 5,000 points
    n_scatter = min(5000, N)
    x_s = x[:n_scatter]
    y_s = y[:n_scatter]
    inside_s = inside[:n_scatter]

    ax_left.scatter(
        x_s[inside_s], y_s[inside_s], s=2, c="tab:blue", alpha=0.6, label="Inside"
    )
    ax_left.scatter(
        x_s[~inside_s], y_s[~inside_s], s=2, c="tab:orange", alpha=0.6, label="Outside"
    )
    theta = np.linspace(0, np.pi / 2, 200)
    ax_left.plot(np.cos(theta), np.sin(theta), c="black", lw=1.5)
    ax_left.set_aspect("equal", adjustable="box")
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("y")
    ax_left.set_title("First 5,000 samples")
    ax_left.legend(loc="upper right", markerscale=5)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.grid(True, linestyle="--", alpha=0.4)

    # Running estimates
    # Compute cumulative mean of inside indicator, then pi_hat running = 4 * cumsum / (1..N)
    inside_float = inside.astype(float)
    cumsum_inside = np.cumsum(inside_float)
    indices = np.arange(1, N + 1)
    running_pi = 4.0 * cumsum_inside / indices
    running_se = 4.0 * np.sqrt(
        (running_pi / 4.0) * (1.0 - running_pi / 4.0) / indices
    )

    # Plot only ~2000 points spaced logarithmically
    plot_indices = np.unique(np.round(np.logspace(0, np.log10(N), 2000)).astype(int))
    plot_indices = plot_indices[plot_indices <= N]
    plot_indices = np.sort(plot_indices)

    ax_top_right.semilogx(plot_indices, running_pi[plot_indices - 1], c="tab:blue", lw=1)
    ax_top_right.fill_between(
        plot_indices,
        running_pi[plot_indices - 1] - 2 * running_se[plot_indices - 1],
        running_pi[plot_indices - 1] + 2 * running_se[plot_indices - 1],
        color="tab:blue",
        alpha=0.2,
    )
    ax_top_right.axhline(math.pi, color="tab:red", linestyle="--", lw=1.5, label="True π")
    ax_top_right.set_xlabel("Number of samples")
    ax_top_right.set_ylabel("Running π estimate")
    ax_top_right.set_title("Running estimate of π")
    ax_top_right.spines["top"].set_visible(False)
    ax_top_right.spines["right"].set_visible(False)
    ax_top_right.grid(True, linestyle="--", alpha=0.4, which="both")
    ax_top_right.legend()

    # 3. Bottom right: |pi_hat - pi| vs N on log-log
    abs_diff = np.abs(running_pi - math.pi)
    # Reference line sigma / sqrt(N) where sigma = 4*sqrt(pi/4 * (1-pi/4))
    sigma_ref = 4.0 * math.sqrt(math.pi / 4.0 * (1.0 - math.pi / 4.0))
    ref_line = sigma_ref / np.sqrt(indices)

    # Use the same log-spaced indices for plotting
    ax_bottom_right.loglog(plot_indices, abs_diff[plot_indices - 1], c="tab:green", lw=1)
    ax_bottom_right.loglog(
        plot_indices, ref_line[plot_indices - 1], c="tab:purple", linestyle="--", lw=1.5, label="σ / √N"
    )
    ax_bottom_right.set_xlabel("Number of samples")
    ax_bottom_right.set_ylabel("|π̂ − π|")
    ax_bottom_right.set_title("Absolute error vs. N")
    ax_bottom_right.spines["top"].set_visible(False)
    ax_bottom_right.spines["right"].set_visible(False)
    ax_bottom_right.grid(True, linestyle="--", alpha=0.4, which="both")
    ax_bottom_right.legend()

    fig.suptitle(
        f"Monte Carlo π = {pi_hat:.5f} ± {se:.5f}  (N = {N:,}, seed = {seed})",
        fontsize=14,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
