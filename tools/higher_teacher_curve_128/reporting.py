"""Complete trajectory with explicit populations and fixed controls."""
from pathlib import Path
import os
import subprocess
from . import common as c


def _training_probe(report):
    if report["diagnostic_scope"] == "full_train":
        result = report["metrics_by_slice"]["original_168_train"]
    else:
        c.require(report["diagnostic_scope"] == "original_168_train_probe", "Unknown TRAIN scope")
        result = report
    c.require(result["metrics"]["all_rows"]["rows"] == 168, "Changed training probe")
    return result


def write_report(binding, result, parent, base=c.OUTPUT):
    base = Path(base)
    paths = {name: base/name for name in ("report.md", "learning_curve.png", "learning_curve.pdf")}
    c.require(all(not path.exists() for path in paths.values()), "Preserve report evidence")
    stats = result["statistics"]
    c.require(stats["checkpoints"] == list(c.CHECKPOINTS), "Incomplete report curve")
    training = {str(f): [c.verified(parent["training"][str(f)][str(u)]) for u in c.CHECKPOINTS] for f in c.FITS}
    probes = {fit: [_training_probe(r) for r in records] for fit, records in training.items()}
    accuracy = [[r["metrics"]["all_rows"]["first_winner_accuracy"]*100 for r in probes[str(f)]] for f in c.FITS]
    capture = [[r["regret_capture_vs_initializer"]*100 for r in probes[str(f)]] for f in c.FITS]
    validation = result["full_validation_old_labels"]
    c.require(set(validation) == {str(u) for u in c.CHECKPOINTS}
              and all(r["states"] == 2489 and r["matched_source_fit"] for r in validation.values()), "Incomplete full VAL curve")
    low = c.verified(parent["validation_old_targets"])["families"]["lowrate"]["metrics"]["all_rows"]
    names = ("focused_outsider_minus_anchor_outsider", "focused_supported_minus_anchor")
    other_names = ("lowrate_outsider_minus_anchor_outsider", "lowrate_supported_minus_anchor")
    references = {name: {
        "lowrate": stats["curves"]["0"]["groups"]["validation"]["contrasts"][other]["mean_bounds"],
        "finalrate": stats["curves"]["1784"]["groups"]["validation"]["contrasts"][name]["mean_bounds"],
    } for name, other in zip(names, other_names)}
    render_input = base/"figure_data.json"
    c.old.save_new(render_input, {"manifest": binding, "statistics": stats, "fits": list(c.FITS),
        "accuracy": accuracy, "validation": validation, "lowrate_validation": low,
        "reference_gains": references, "training_probe_rows": 168, "full_validation_rows": 2489})
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with (base/"render_stdout.txt").open("x", encoding="utf-8") as out, (base/"render_stderr.txt").open("x", encoding="utf-8") as err:
        rendered = subprocess.run([str(c.windows._windowless_executable()), "-u",
            str(c.ROOT/"tools/higher_teacher_curve_128/render.py"), str(render_input)], cwd=c.ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    c.require(rendered.returncode == 0 and all(paths[n].is_file() for n in ("learning_curve.png", "learning_curve.pdf")),
              "Chart rendering failed; inspect saved logs")
    receipt = c.old.read_json(base/"render_complete.json")
    c.require(receipt["manifest"] == binding and receipt["input_sha256"] == c.old.file_hash(render_input)
              and receipt["torch_imported"] is False
              and all(receipt["figures"][n] == c.old.file_hash(paths[n]) for n in ("learning_curve.png", "learning_curve.pdf")),
              "Chart input identity changed")
    lines = ["# Higher-rate teacher learning curve", "",
        "All 13 saved stages and all three models were retained. All new predictions, private-information checks, "
        "old-label metrics, decisions, arithmetic and paired statistics passed independent verification. "
        "No new training, source games or simulations were performed. No checkpoint was selected.", "",
        "The models share 9,461 TRAIN positions from 159 pairs. Full validation diagnostics use 2,489 positions "
        "from 42 other pairs, with each position's assigned model. Conditional planning uses 336 positions: "
        "168 TRAIN and 168 VAL, 42 pairs each. These are reused development outcomes, not fresh confirmation or game win rates.", "",
        f"![Saved learning trajectory]({paths['learning_curve.png'].resolve().as_posix()})", "",
        "## Learning and generalization", "",
        "TRAIN columns consistently use the original 168-position probe, averaged across all three models. "
        "Full VAL uses the assigned model per row. Agreement excludes empty outsider sets; loss and regret retain all rows.", "",
        "| Update | TRAIN probe agreement | TRAIN probe regret removed | Full VAL agreement | Full VAL tied-best | Full VAL loss | Full VAL regret pp |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for i, u in enumerate(c.CHECKPOINTS):
        m = validation[str(u)]["metrics"]["all_rows"]
        lines.append(f"| {u} | {sum(row[i] for row in accuracy)/3:.2f}% | {sum(row[i] for row in capture)/3:.2f}% | "
            f"{m['first_winner_accuracy']*100:.2f}% | {m['any_old_max_accuracy']*100:.2f}% | "
            f"{m['mean_loss']:.5f} | {m['mean_old_score_regret']*100:.3f} |")
    lines += ["", f"Fixed lower-rate full VAL reference: agreement {low['first_winner_accuracy']*100:.2f}%, "
        f"tied-best {low['any_old_max_accuracy']*100:.2f}%, loss {low['mean_loss']:.5f}, "
        f"regret {low['mean_old_score_regret']*100:.3f}pp.", "",
        "## Full training endpoints", "", "Each model is measured on all 9,461 TRAIN rows here, separately from the probe curve.", "",
        "| Model | Initial agreement | Final agreement | Final tied-best | Final loss | Old regret removed |",
        "|---:|---:|---:|---:|---:|---:|"]
    for f in c.FITS:
        first, last = training[str(f)][0], training[str(f)][-1]
        m = last["metrics"]["all_rows"]
        c.require(first["metrics"]["all_rows"]["rows"] == m["rows"] == 9461, "Wrong full TRAIN endpoint")
        lines.append(f"| {f} | {first['metrics']['all_rows']['first_winner_accuracy']*100:.2f}% | "
            f"{m['first_winner_accuracy']*100:.2f}% | {m['any_old_max_accuracy']*100:.2f}% | "
            f"{m['mean_loss']:.5f} | {last['regret_capture_vs_initializer']*100:.2f}% |")
    for role, title, suffixes in (
        ("supported", "Validation planning", ("anchor", "lowrate_supported", "finalrate_supported")),
        ("outsider", "Validation outsider quality", ("anchor_outsider", "lowrate_outsider", "finalrate_outsider")),
    ):
        lines += ["", "## "+title, "", "Values are percentage points; brackets are simultaneous 95% bands.", "",
            "| Update | Versus anchor | Versus fixed lower rate | Versus fixed final higher rate |", "|---:|---:|---:|---:|"]
        for u in c.CHECKPOINTS:
            cells = []
            for suffix in suffixes:
                name = f"focused_{role}_minus_{suffix}"
                row = stats["curves"][str(u)]["groups"]["validation"]["contrasts"][name]
                point = f"{row['mean']*100:+.3f}" if row["mean"] is not None else f"[{row['mean_bounds'][0]*100:+.3f}, {row['mean_bounds'][1]*100:+.3f}]"
                if u:
                    band = stats["simultaneous_validation"]["groups"][str(u)][name]["ci95"]
                    point += f" [{band[0]*100:+.3f}, {band[1]*100:+.3f}]"
                else:
                    point += " (initial reference)"
                cells.append(point)
            lines.append(f"| {u} | "+" | ".join(cells)+" |")
    lines += ["", "## Interpretation limits", "",
        "One approximate 95% family covers 72 noninitial stage-by-comparison cells: 12 stages, two decision types "
        "and three fixed references. It uses 50,000 shared whole-pair bootstrap draws with fixed-standard-error "
        "maximum centered deviations. Checkpoints and model repetitions do not increase the independent pair count. "
        "This correction does not undo earlier use of the development positions.", "",
        "Beating the anchor alone does not establish superiority to either completed model. Better old-target loss "
        "alone does not establish better planning. A credible intermediate region motivates a separately fixed rule "
        "and fresh full-legal confirmation. No clear region reduces the priority of further checkpoint or optimizer "
        "tuning for this trajectory; wide intervals still leave modest gains unresolved. No promotion, checkpoint "
        "selection or automatic successor follows.", "", f"Frozen binding: {binding}", ""]
    with paths["report.md"].open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    return {"report": c.artifact(paths["report.md"]), "figure": c.artifact(paths["learning_curve.png"]),
        "figure_pdf": c.artifact(paths["learning_curve.pdf"]), "figure_data": c.artifact(render_input),
        "render_receipt": c.artifact(base/"render_complete.json"),
        "render_logs": [c.artifact(base/n) for n in ("render_stdout.txt", "render_stderr.txt")]}
