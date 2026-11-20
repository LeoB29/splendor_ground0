"""One saved-result figure; no search, training, or live evidence access."""
from pathlib import Path


def render(report, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    cells = {(c["quota"], c["scope"]): c for c in report["cells"]}
    colors = {"3": "#3568B0", "6": "#D47E24"}
    positions = np.arange(2, dtype=float)
    matched = {str(c["pool"]): c for c in report["matched_budget_contrasts"] if c["scope"] == "legacy_total"}
    paths = []
    for same_roots in (False, True):
        figure, axes = plt.subplots(1, 2, figsize=(11.4, 5.7), dpi=160)
        for ax, metric, factor, ylabel, title in (
            (axes[0], "jsd", 1., "Distribution disagreement (JSD, nats)", "How much do move preferences vary?"),
            (axes[1], "action_disagreement", 100., "Pairs choosing different moves (%)", "How often does the chosen move change?"),
        ):
            for i, pool in enumerate(("3", "6")):
                values = ([matched[pool]["metrics"][metric]["estimates"][str(q)] for q in (512, 2048)] if same_roots else
                          [cells[(q, "legacy_total")]["metrics"][metric]["estimates"][pool] for q in (512, 2048)])
                means = np.asarray([v["mean"] * factor for v in values])
                lower = np.asarray([v["ci95"][0] * factor for v in values])
                upper = np.asarray([v["ci95"][1] * factor for v in values])
                x = positions + (i - .5) * .27
                # Percentile intervals need not contain the point estimate.
                ax.vlines(x, lower, upper, linewidth=2, color=colors[pool])
                ax.hlines(lower, x - .035, x + .035, linewidth=2, color=colors[pool])
                ax.hlines(upper, x - .035, x + .035, linewidth=2, color=colors[pool])
                ax.scatter(x, means, s=50, color=colors[pool], label=pool + " worlds", zorder=3)
                if same_roots: ax.plot(x, means, color=colors[pool], linewidth=1, alpha=.55)
            counts = (64, 64) if same_roots else (128, 64)
            ax.set_xticks(positions, (f"512 simulations\n{counts[0]} saved positions", f"2,048 simulations\n{counts[1]} saved positions"))
            ax.set_xlim(-.45, 1.45)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, pad=12)
            ax.grid(axis="y", color="#DDDDDD", linewidth=.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=10)
        figure.suptitle("Effect of more search on the same 64 saved positions" if same_roots else
                        "Teacher repeatability: three versus six hidden-card worlds", fontsize=14, y=.97)
        note = ("Both budgets use identical positions; all budget contrasts are exploratory." if same_roots else
                "The two budgets use different position sets; use the matched-budget chart for depth comparisons.")
        figure.text(.5, .035,
                    "Lower means more repeatable, not necessarily stronger. Four independent legacy seeds per setting.\n"
                    "Points are equal-game means; bars are game-bootstrap 95% intervals.\n" + note,
                    ha="center", va="bottom", fontsize=9, color="#444444")
        figure.subplots_adjust(left=.08, right=.97, bottom=.27, top=.83, wspace=.30)
        path = destination / ("teacher_variability_matched_budget.png" if same_roots else "teacher_variability_summary.png")
        figure.savefig(path, facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths
