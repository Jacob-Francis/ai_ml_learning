"""
Week 2, tasks 6-7: fit a power law L(N) = L_inf + A*N^-alpha to the 4 scaling runs, and compute
the compute-optimal frontier across every logged (FLOPs, val_loss) point from all 4 runs.

Run with: python fit_scaling_law.py (after run_all.py has produced results/scaling_table.json
and every results/<run_name>/loss_history.json). Writes results/scaling_law_fit.json (both fits'
parameters + CIs + the df=1 caveat text) and results/compute_frontier.png.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t as student_t

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

from flops import load_run

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# --- The model being fit ---

def power_law(N, L_inf, A, alpha):
    """L(N) = L_inf + A * N^-alpha - loss approaches an irreducible floor L_inf as N -> infinity,
    with a power-law-shaped excess loss A*N^-alpha on top for finite N. This is the exact form
    Kaplan et al. 2020 fit (their eq. 1.1), which is why alpha here is directly comparable to
    their reported ~0.076 (same functional form, same meaning: how fast excess loss shrinks as
    you 10x the parameter count)."""
    return L_inf + A * np.power(N, -alpha)


def fit_full_power_law(N, L):
    """
    Nonlinear least squares fit of the full 3-parameter power law via scipy.optimize.curve_fit.

    Why curve_fit and not a log-log linear fit: L(N) = L_inf + A*N^-alpha is *not* linearizable
    in log-log space when L_inf != 0 - taking log(L) doesn't turn the additive L_inf term into
    something a straight line can absorb (log(L_inf + A*N^-alpha) != log(L_inf) + log(A*N^-alpha)
    in general). So this needs genuine nonlinear least squares, not OLS on logs (that's what the
    *second*, simpler fit below is for, with L_inf pinned to 0 specifically so it CAN be
    linearized).

    Initial guesses (deliberately not just "start at Kaplan's alpha=0.076" - that would bias the
    fit toward confirming the literature value rather than actually measuring ours):
      - alpha0 = 0.5: a middle-of-the-road guess with no literature-value bias, i.e. genuinely
        uninformed about which regime this sweep will land in.
      - L_inf0 = 0.9 * min(L): assume the irreducible floor is a bit below the best loss we've
        actually observed (since with only 4 finite-N points, the true infinite-N floor should be
        lower than the smallest-N-so-far loss, but probably not by a huge margin at this scale).
      - A0: back-solved from the smallest-N point (N is smallest -> excess loss A*N^-alpha is
        largest there, so it's the best-conditioned point to invert for A):
            L[0] = L_inf0 + A0 * N[0]^-alpha0
            =>  A0 = (L[0] - L_inf0) * N[0]^alpha0

    Returns:
        dict with alpha, alpha_se, alpha_ci95 (from the covariance matrix - see the df=1 caveat
        below), L_inf, A, and the raw fitted params/covariance for reference.
    """
    N = np.asarray(N, dtype=float)
    L = np.asarray(L, dtype=float)

    # smallest N first, so N[0]/L[0] below really are the smallest-N point
    order = np.argsort(N)
    N, L = N[order], L[order]

    alpha0 = 0.5
    L_inf0 = 0.9 * L.min()
    A0 = (L[0] - L_inf0) * N[0] ** alpha0

    p0 = [L_inf0, A0, alpha0]
    # Bounds: L_inf and A should be non-negative (a negative irreducible loss floor or a negative
    # excess-loss coefficient would both be nonsensical for a cross-entropy loss); alpha is left
    # unbounded above/below zero since a priori we don't know its sign here (a *negative* alpha,
    # i.e. loss getting worse with more params, isn't physically expected but isn't ruled out by
    # the fit procedure itself - if that's what 4 noisy points imply, better to see it than to
    # have masked it with an overly tight bound).
    bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])

    popt, pcov = curve_fit(power_law, N, L, p0=p0, bounds=bounds, maxfev=20000)
    L_inf_fit, A_fit, alpha_fit = popt

    # Standard errors from the diagonal of the covariance matrix curve_fit returns.
    perr = np.sqrt(np.diag(pcov))
    alpha_se = perr[2]

    # --- The df=1 caveat ---
    # 4 data points, 3 free parameters (L_inf, A, alpha) -> degrees of freedom = 4 - 3 = 1.
    # A 95% CI half-width is normally computed as ~1.96 * SE (the z-score, valid as
    # n -> infinity), but with only 1 residual degree of freedom the correct multiplier is the
    # t-distribution's 97.5th percentile at df=1, which is much fatter-tailed:
    t_mult = student_t.ppf(0.975, df=1)  # ~= 12.7, vs. 1.96 for a normal/large-sample CI
    alpha_ci95_halfwidth = t_mult * alpha_se
    alpha_ci95 = (alpha_fit - alpha_ci95_halfwidth, alpha_fit + alpha_ci95_halfwidth)

    caveat = (
        f"With 4 data points and 3 free parameters (L_inf, A, alpha), there is exactly 1 "
        f"residual degree of freedom. The 95% CI here uses the t-distribution at df=1 "
        f"(t.ppf(0.975, df=1)={t_mult:.2f}), not the usual normal-approximation 1.96x - about "
        f"{t_mult:.1f}x wider than a naive large-sample CI would report. This means alpha is "
        f"poorly constrained *by design*, not that anything went wrong: with only 4 points, "
        f"there's essentially no slack left to independently estimate uncertainty once 3 "
        f"parameters are fit to it. Take alpha's point estimate as a rough indication of trend, "
        f"not a tightly pinned-down exponent."
    )

    return {
        "L_inf": float(L_inf_fit),
        "A": float(A_fit),
        "alpha": float(alpha_fit),
        "alpha_se": float(alpha_se),
        "alpha_ci95": [float(alpha_ci95[0]), float(alpha_ci95[1])],
        "t_multiplier_df1": float(t_mult),
        "degrees_of_freedom": 1,
        "caveat": caveat,
        "p0": {"L_inf0": float(L_inf0), "A0": float(A0), "alpha0": float(alpha0)},
        "popt": [float(x) for x in popt],
        "pcov": pcov.tolist(),
    }


def fit_simple_power_law(N, L):
    """
    Sanity-check fit: pin L_inf=0, i.e. fit the pure power law L(N) = A*N^-alpha. This *is*
    linearizable: log(L) = log(A) - alpha*log(N), a straight line in log-log space, so it can be
    fit with plain ordinary least squares (np.polyfit on the logs) instead of nonlinear
    optimization - a useful independent check that isn't relying on curve_fit's initial guess or
    bounds at all. 2 free parameters (log(A), alpha) instead of 3, so with 4 points this has 2
    degrees of freedom - still few, but less extreme than the full fit's 1.

    Returns:
        dict with alpha, A, and the OLS fit's R^2 (how well a pure power law - no floor term -
        explains these 4 points on its own).
    """
    N = np.asarray(N, dtype=float)
    L = np.asarray(L, dtype=float)

    log_N = np.log(N)
    log_L = np.log(L)

    # np.polyfit(x, y, 1) fits y = m*x + b by OLS; here y=log(L), x=log(N), so m = -alpha,
    # b = log(A).
    m, b = np.polyfit(log_N, log_L, 1)
    alpha = -m
    A = np.exp(b)

    # R^2 of the log-log linear fit, as a rough "does a pure power law even look right" check.
    residuals = log_L - (m * log_N + b)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_L - log_L.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"alpha": float(alpha), "A": float(A), "r_squared": float(r_squared),
            "degrees_of_freedom": len(N) - 2}


def compute_frontier(all_points):
    """
    Pool every (flops, val_loss) point from every run and take the lower envelope: sorted by
    FLOPs ascending, a point only joins the frontier if its loss beats every point already on the
    frontier (i.e. it's a new all-time-low loss for the FLOPs spent so far). This traces out
    "the best loss achievable for a given compute budget, using the best of these 4 model sizes
    at that budget" - the empirical compute-optimal frontier.

    Args:
        all_points: list of (run_name, flops, loss) tuples, pooled across every run's *val*
            loss_history entries (not just the 4 final/best points - every epoch's val
            checkpoint from every run contributes a candidate point).

    Returns:
        frontier: list of (run_name, flops, loss) tuples forming the lower envelope, sorted by
            ascending flops.
    """
    pts = sorted(all_points, key=lambda p: p[1])  # sort by flops ascending
    frontier = []
    best_loss_so_far = float("inf")
    for run_name, flops, loss in pts:
        if loss < best_loss_so_far:
            frontier.append((run_name, flops, loss))
            best_loss_so_far = loss
    return frontier


def main():
    table = json.loads((RESULTS_DIR / "scaling_table.json").read_text())

    N_vals = [r["n_params"] for r in table]
    L_vals = [r["final_val_loss"] for r in table]

    print("Input (N, best val loss) points:")
    for r in table:
        print(f"  {r['run_name']:<12} N={r['n_params']:>10,}  L={r['final_val_loss']:.4f}")

    full_fit = fit_full_power_law(N_vals, L_vals)
    simple_fit = fit_simple_power_law(N_vals, L_vals)

    print(f"\nFull fit  (L_inf + A*N^-alpha): alpha={full_fit['alpha']:.4f} "
          f"+/- {full_fit['alpha_se']:.4f} (SE), 95% CI={tuple(round(x, 4) for x in full_fit['alpha_ci95'])}")
    print(f"  L_inf={full_fit['L_inf']:.4f}, A={full_fit['A']:.4g}")
    print(f"  {full_fit['caveat']}")
    print(f"\nSimple fit (pure power law, L_inf=0): alpha={simple_fit['alpha']:.4f}, "
          f"A={simple_fit['A']:.4g}, R^2={simple_fit['r_squared']:.4f} "
          f"(df={simple_fit['degrees_of_freedom']})")
    print(f"\nFor reference, Kaplan et al. (2020) report alpha ~= 0.076 for their token-level "
          f"BPE-tokenized web-text sweep.")

    # --- Compute-optimal frontier: pool every val point from every run ---
    all_points = []  # (run_name, flops, loss)
    for r in table:
        run_dir = RESULTS_DIR / r["run_name"]
        loaded = load_run(run_dir)
        for step, loss, flops in loaded["val"]:
            all_points.append((r["run_name"], flops, loss))

    frontier = compute_frontier(all_points)
    print(f"\nCompute-optimal frontier: {len(frontier)} points out of {len(all_points)} pooled "
          f"val checkpoints across all 4 runs.")
    for run_name, flops, loss in frontier:
        print(f"  {run_name:<12} flops={flops:.3e}  loss={loss:.4f}")

    # --- Save the fit results (loaded by the notebook, so it doesn't need scipy at all if it
    #     just wants to show numbers, though it re-plots using these saved values) ---
    fit_results = {
        "n_points": len(N_vals),
        "N": N_vals,
        "L": L_vals,
        "run_names": [r["run_name"] for r in table],
        "full_fit": full_fit,
        "simple_fit": simple_fit,
        "kaplan_alpha_reference": 0.076,
        "frontier": [{"run_name": rn, "flops": f, "loss": l} for rn, f, l in frontier],
    }
    fit_path = RESULTS_DIR / "scaling_law_fit.json"
    fit_path.write_text(json.dumps(fit_results, indent=2))
    print(f"\nSaved {fit_path}")

    # --- Plot 1: L(N) fit ---
    plt.figure(figsize=(7, 5))
    N_arr = np.array(N_vals, dtype=float)
    L_arr = np.array(L_vals, dtype=float)
    plt.scatter(N_arr, L_arr, color="#2a78d6", zorder=3, label="observed (N, best val loss)")
    N_smooth = np.geomspace(N_arr.min() * 0.7, N_arr.max() * 1.3, 200)
    plt.plot(N_smooth, power_law(N_smooth, *full_fit["popt"]), color="#eb6834",
              label=f"fit: L_inf={full_fit['L_inf']:.3f}, alpha={full_fit['alpha']:.3f}"
                    f" (95% CI [{full_fit['alpha_ci95'][0]:.2f}, {full_fit['alpha_ci95'][1]:.2f}], df=1)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N (parameters, log scale)")
    plt.ylabel("best validation loss (log scale)")
    plt.title("Scaling law fit: L(N) = L_inf + A*N^-alpha")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fit_plot_path = RESULTS_DIR / "scaling_law_fit.png"
    plt.savefig(fit_plot_path)
    plt.close()
    print(f"Saved {fit_plot_path}")

    # --- Plot 2: compute-optimal frontier ---
    RUN_COLORS = {
        "scale_100k": "#2a78d6",
        "scale_250k": "#eb6834",
        "scale_500k": "#1baf7a",
        "scale_1m": "#eda100",
    }
    plt.figure(figsize=(7.5, 5.5))
    for r in table:
        run_name = r["run_name"]
        run_dir = RESULTS_DIR / run_name
        loaded = load_run(run_dir)
        # loaded["val"] is a list of (step, loss, flops) triples (see flops.load_run) - pull out
        # just the flops/loss columns for this run's scatter points.
        flops_pts = [flops for (_step, _loss, flops) in loaded["val"]]
        loss_pts = [loss for (_step, loss, _flops) in loaded["val"]]
        color = RUN_COLORS.get(run_name, "#4a3aa7")
        plt.scatter(flops_pts, loss_pts, color=color, alpha=0.6, s=18, label=run_name)

    frontier_flops = [p["flops"] for p in fit_results["frontier"]]
    frontier_loss = [p["loss"] for p in fit_results["frontier"]]
    plt.step(frontier_flops, frontier_loss, where="post", color="black", linewidth=2.2,
              label="compute-optimal frontier", zorder=5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("training FLOPs (6*N*tokens, log scale)")
    plt.ylabel("validation loss (log scale)")
    plt.title("Loss vs. compute, all runs, with compute-optimal frontier")
    plt.legend(fontsize=8)
    plt.tight_layout()
    frontier_plot_path = RESULTS_DIR / "compute_frontier.png"
    plt.savefig(frontier_plot_path)
    plt.close()
    print(f"Saved {frontier_plot_path}")


if __name__ == "__main__":
    main()
