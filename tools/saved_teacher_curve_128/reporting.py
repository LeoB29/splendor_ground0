"""Readable complete trajectory and exportable scientific figure."""
from pathlib import Path
import os
import subprocess
from . import common as c


def write_report(binding, result, parent, base=c.OUTPUT):
    base = Path(base)
    paths = {name: base/name for name in ("report.md", "learning_curve.png", "learning_curve.pdf")}
    c.require(all(not path.exists() for path in paths.values()), "Preserve report evidence")
    stats = result["statistics"]
    c.require(stats["checkpoints"] == list(c.CHECKPOINTS), "Incomplete report curve")
    x = c.CHECKPOINTS
    train = {str(fit): [c.verified(parent["training"][str(fit)][str(update)]) for update in x] for fit in c.FITS}
    accuracy = [[row["metrics"]["all_rows"]["first_winner_accuracy"]*100 for row in train[str(fit)]] for fit in c.FITS]
    capture = [[row["regret_capture_vs_initializer"]*100 for row in train[str(fit)]] for fit in c.FITS]
    names = ("focused_outsider_minus_anchor_outsider", "focused_supported_minus_anchor")
    render_input = base/"figure_data.json"
    c.old.save_new(render_input, {"manifest": binding, "statistics": stats, "fits": list(c.FITS), "accuracy": accuracy})
    # A separate native runtime avoids Torch and Matplotlib loading competing
    # Intel OpenMP libraries into one process. No numerical workaround is used.
    render_env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with (base/"render_stdout.txt").open("x", encoding="utf-8") as out, (base/"render_stderr.txt").open("x", encoding="utf-8") as err:
        rendered = subprocess.run([str(c.windows._windowless_executable()), "-u",
                                   str(c.ROOT/"tools/saved_teacher_curve_128/render.py"), str(render_input)],
                                  cwd=c.ROOT, env=render_env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=120,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    c.require(rendered.returncode == 0 and all(paths[name].is_file() for name in ("learning_curve.png", "learning_curve.pdf")),
              "Chart rendering failed; inspect saved rendering logs")
    render_receipt = c.old.read_json(base/"render_complete.json")
    c.require(render_receipt["manifest"] == binding and render_receipt["input_sha256"] == c.old.file_hash(render_input)
              and render_receipt["torch_imported"] is False
              and all(render_receipt["figures"][name] == c.old.file_hash(paths[name]) for name in ("learning_curve.png", "learning_curve.pdf")),
              "Chart input identity changed")
    lines = ["# Saved teacher learning curve", "",
             "All 13 saved stages from all three models were assessed on the same 336 positions. "
             "The study used existing checkpoints and independently generated assessment outcomes, with no new training or simulations. "
             "All new predictions, choices, arithmetic and statistics passed independent verification.", "",
             "Each split contains 168 positions from 42 complete source pairs. The models share the same training set. "
             "These are reused development results; validation positions and outcomes have been inspected before. "
             "Conditional terminal-score changes do not establish complete-game strength.", "",
             f"![Complete learning curve]({paths['learning_curve.png'].as_posix()})", "",
             "## Every saved stage", "",
             "Agreement and regret capture below average the three full training histories. "
             "Utility uses each position's assigned model. Validation intervals simultaneously cover "
             "both primary endpoints at all 12 noninitial stages. No model or checkpoint is selected.", "",
             "| Update | TRAIN agreement | TRAIN regret removed | TRAIN outsider gain | VALIDATION outsider gain [95% band] | VALIDATION planning gain [95% band] |",
             "|---:|---:|---:|---:|---:|---:|"]
    for j, update in enumerate(x):
        def entry(name, split):
            return stats["curves"][str(update)]["groups"][split]["contrasts"][name]
        def point(row):
            return f"{row['mean']*100:+.3f}" if row["mean"] is not None else f"[{row['mean_bounds'][0]*100:+.3f}, {row['mean_bounds'][1]*100:+.3f}]"
        def val(name):
            value = point(entry(name, "validation"))
            if update == 0: return value + " (baseline)"
            band = stats["simultaneous_validation"]["groups"][str(update)][name]["ci95"]
            return value + f" [{band[0]*100:+.3f}, {band[1]*100:+.3f}]"
        lines.append(f"| {update} | {sum(row[j] for row in accuracy)/len(accuracy):.2f}% | {sum(row[j] for row in capture)/len(capture):.2f}% | "
                     f"{point(entry(names[0], 'train'))} | {val(names[0])} | {val(names[1])} |")
    lines += ["", "All utility values and bands are percentage points. Agreement excludes empty outsider sets; "
              "utility retains them as structural zero differences. The JSON assessment includes individual model histories, "
              "old-target metrics, all contextual contrasts and descriptive pointwise intervals.", "",
              "## Agreement with old teaching targets", "",
              "This table uses each position's assigned model in both splits. Its TRAIN agreement therefore has a different "
              "aggregation from the average of all three full training histories above. Cross-entropy includes zero-loss "
              "empty and singleton supports in the mean; regret includes every position. Neither metric selects a checkpoint.", "",
              "| Update | TRAIN agreement | VALIDATION agreement | TRAIN loss | VALIDATION loss | TRAIN old regret pp | VALIDATION old regret pp |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for update in x:
        groups = result["old_label_metrics"][str(update)]["groups"]
        train_metric = groups["train"]["metrics"]["all_rows"]
        val_metric = groups["validation"]["metrics"]["all_rows"]
        lines.append(f"| {update} | {100*train_metric['first_winner_accuracy']:.2f}% | "
                     f"{100*val_metric['first_winner_accuracy']:.2f}% | "
                     f"{train_metric['mean_cross_entropy']:.5f} | {val_metric['mean_cross_entropy']:.5f} | "
                     f"{100*train_metric['mean_old_score_regret']:.3f} | {100*val_metric['mean_old_score_regret']:.3f} |")
    lines += ["",
              "The simultaneous bands use 50,000 shared whole-pair bootstrap draws within fit and split, "
              "with a maximum centered deviation standardized by each endpoint's fixed whole-pair standard error. "
              "They address the declared checkpoint scan; "
              "they do not remove the limitations of reused development data or the fixed training set.", "",
              "An intermediate positive region would motivate a separately fixed rule and fresh confirmation. "
              "Outsider improvement alone does not establish planning improvement. Wide bands leave transfer unresolved. "
              "No automatic successor or promotion follows.", "",
              f"Frozen study binding: {binding}", ""]
    with paths["report.md"].open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    return {"report": c.artifact(paths["report.md"]), "figure": c.artifact(paths["learning_curve.png"]),
            "figure_pdf": c.artifact(paths["learning_curve.pdf"]), "figure_data": c.artifact(render_input),
            "render_receipt": c.artifact(base/"render_complete.json"),
            "render_logs": [c.artifact(base/name) for name in ("render_stdout.txt", "render_stderr.txt")]}
