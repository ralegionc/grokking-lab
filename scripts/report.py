"""Collect every finished run into one markdown table.

Numbers in the README come from here rather than from anyone's memory. Run it
after training and it rewrites RESULTS.md in place.

    python scripts/report.py --runs runs --out RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COLUMNS = [
    ("run", lambda s: s["run_name"]),
    ("task", lambda s: f"{s['op']} mod {s['p']}"),
    ("model", lambda s: s["arch"]),
    ("train fraction", lambda s: f"{s['train_frac']:g}"),
    ("weight decay", lambda s: f"{s['weight_decay']:g}"),
    ("memorised", lambda s: _step(s["memorisation_step"])),
    ("generalised", lambda s: _step(s["grok_step_95"])),
    ("gap", lambda s: _gap(s)),
    ("final held out", lambda s: f"{s['final_val_acc']:.3f}"),
    ("key frequencies", lambda s: ", ".join(str(f) for f in s["key_frequencies"]) or "none"),
]


def _step(value) -> str:
    return "never" if value is None else f"{value:,}"


def _gap(summary) -> str:
    memo, grok = summary["memorisation_step"], summary["grok_step_95"]
    if memo is None or grok is None:
        return "-"
    if memo == 0:
        return f"{grok:,}"
    return f"{grok - memo:,} steps, {grok / max(memo, 1):.0f}x"


def collect(root: Path, exclude: list[str] | None = None) -> list[dict]:
    """Every run under root, optionally skipping whole subdirectories.

    Skipping is how the README gets a five row table while RESULTS.md keeps all
    twenty one: the sweep points matter in aggregate, not individually.
    """
    exclude = exclude or []
    summaries = []
    for path in sorted(root.rglob("summary.json")):
        if any(part in exclude for part in path.parts):
            continue
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def to_markdown(summaries: list[dict]) -> str:
    head = "| " + " | ".join(name for name, _ in COLUMNS) + " |"
    rule = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    rows = [
        "| " + " | ".join(render(summary) for _, render in COLUMNS) + " |"
        for summary in summaries
    ]
    return "\n".join([head, rule, *rows])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python scripts/report.py")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("RESULTS.md"))
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="DIR",
        help="directory names to skip, e.g. --exclude sweeps",
    )
    args = parser.parse_args(argv)

    summaries = collect(args.runs, args.exclude)
    if not summaries:
        raise SystemExit(f"no summary.json found under {args.runs}")

    total = sum(s.get("wall_seconds", 0) for s in summaries)
    body = [
        "# Results",
        "",
        "Written by `scripts/report.py` from the summary of every run under `runs/`.",
        f"{len(summaries)} runs, {total / 3600:.1f} hours of CPU time in total.",
        "",
        "`memorised` is the first evaluation at which training accuracy reached 99%.",
        "`generalised` is the first at which held out accuracy reached 95%.",
        "",
        to_markdown(summaries),
        "",
    ]
    args.out.write_text("\n".join(body), encoding="utf-8")
    print("\n".join(body))


if __name__ == "__main__":
    main()
