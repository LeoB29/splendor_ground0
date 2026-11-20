"""Standalone chart renderer. Deliberately imports no Torch or project module."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(request):
    request = Path(request).resolve()
    raw = request.read_bytes()
    payload = json.loads(raw)
    stats, fits = payload["statistics"], payload["fits"]
    x = np.array(stats["checkpoints"])
    accuracy = np.array(payload["accuracy"])
    assert accuracy.shape == (3, 13) and len(x) == 13 and len(fits) == 3
    paths = {name: request.parent/name for name in ("learning_curve.png", "learning_curve.pdf", "render_complete.json")}
    assert not any(path.exists() for path in paths.values())
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True, constrained_layout=True)
    for index, fit in enumerate(fits):
        axes[0].plot(x, accuracy[index], marker=".", linewidth=1.2, alpha=.65, label=f"TRAIN probe: {fit}")
    validation = [payload["validation"][str(u)]["metrics"]["all_rows"] for u in x]
    axes[0].plot(x, [r["first_winner_accuracy"]*100 for r in validation], color="#bb422e", linewidth=2,
                 marker="o", markersize=3, label="Full VAL: assigned models")
    low = payload["lowrate_validation"]
    axes[0].axhline(low["first_winner_accuracy"]*100, color="#555555", linestyle=":", label="Lower-rate final VAL")
    axes[0].set_ylabel("Teacher agreement (%)")
    axes[0].set_title("Teacher agreement: fixed 168-position TRAIN probe and full 2,489-position VAL", loc="left", fontsize=11)
    axes[0].set_ylim(0, 103)
    axes[0].legend(loc="upper left", ncols=2, fontsize=8)
    axes[1].plot(x, [r["mean_loss"] for r in validation], color="#bb422e", marker="o", markersize=3,
                 label="Higher-rate trajectory")
    axes[1].axhline(low["mean_loss"], color="#555555", linestyle=":", label="Lower-rate final model")
    axes[1].set_title("Full validation loss against old teaching targets (lower is better)", loc="left", fontsize=11)
    axes[1].set_ylabel("Cross-entropy")
    axes[1].legend(loc="best", fontsize=8)
    names = ("focused_outsider_minus_anchor_outsider", "focused_supported_minus_anchor")
    labels = ("Quality of the additional proposed move", "Quality of the resulting planning decision")
    uncertain = False
    for ax, name, label in zip(axes[2:], names, labels):
        for split, color, legend in (("train", "#2563a6", "Training positions"), ("validation", "#bb422e", "Validation positions")):
            entries = [stats["curves"][str(update)]["groups"][split]["contrasts"][name] for update in x]
            uncertain |= any(row["mean"] is None for row in entries)
            centers = np.array([np.mean(row["mean_bounds"]) for row in entries])*100
            ax.plot(x, centers, marker="o", markersize=3, color=color, label=legend)
        entries = [stats["simultaneous_validation"]["groups"][str(update)][name] for update in x[1:]]
        bands = np.array([row["ci95"] for row in entries])*100
        ax.fill_between(x[1:], bands[:, 0], bands[:, 1], color="#bb422e", alpha=.15,
                        label="Validation simultaneous 95% band")
        ax.axhline(0, color="#555555", linewidth=.8, linestyle="--")
        references = payload["reference_gains"][name]
        ax.axhline(np.mean(references["lowrate"])*100, color="#26805b", linewidth=1,
                   linestyle=":", label="Lower-rate final VAL")
        ax.axhline(np.mean(references["finalrate"])*100, color="#7b509e", linewidth=1,
                   linestyle="-.", label="Higher-rate final VAL")
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_ylabel("Score gain vs baseline (pp)")
        ax.legend(loc="best", ncols=2, fontsize=8)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1)
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    ticks = [0, 1, 8, 32, 128, 512, 1784]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([str(value) for value in ticks])
    axes[-1].set_xlim(-.08, 2400)
    axes[-1].set_xlabel("Saved training update (log spacing above 1)")
    fig.suptitle("Does teacher learning transfer beyond the practiced positions?", fontsize=14)
    footer = "Reused development positions and simulation outcomes; these are not game win rates.\n"
    footer += "Validation bands jointly cover 12 noninitial stages x 6 comparisons; no checkpoint selected."
    if uncertain: footer += "\nLines show bound midpoints where outcomes are unresolved; full bounds remain in the report."
    fig.supxlabel(footer, fontsize=8)
    fig.savefig(paths["learning_curve.png"], dpi=170)
    fig.savefig(paths["learning_curve.pdf"])
    plt.close(fig)
    with paths["render_complete.json"].open("x", encoding="utf-8") as handle:
        json.dump({"manifest": payload["manifest"], "input_sha256": hashlib.sha256(raw).hexdigest(),
                   "checkpoints": list(map(int, x)), "fits": fits, "torch_imported": "torch" in sys.modules,
                   "figures": {name: hashlib.sha256(paths[name].read_bytes()).hexdigest()
                               for name in ("learning_curve.png", "learning_curve.pdf")}}, handle, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
