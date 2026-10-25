"""Where the analysis scripts find eval logs and fabrication labels.

`data/` is the tracked release: the exact logs and frozen two-judge labels
behind the paper. `logs/` is gitignored local working output; it can hold
later re-runs under the same cell names (the July/August Grok, Opus and Sonnet
reruns did exactly that), so reading it first would silently change published
numbers. Every lookup therefore tries the release first and falls back to
`logs/` only for cells that were never released.
"""
import glob
import json
import os

LOG_ROOTS = ("data/eval_logs", "logs")
LABEL_FILES = ("logs/_fab_labels.json", "data/fab_labels.json")  # later wins


def log_path(cell):
    """Path of the latest .eval for a cell, or None if no root has it."""
    for root in LOG_ROOTS:
        fs = sorted(glob.glob(f"{root}/{cell}/*.eval"))
        if fs:
            return fs[-1]
    return None


def has_log(cell):
    return log_path(cell) is not None


def samples(cell):
    """Samples of the latest log for a cell; [] if the cell is missing."""
    from inspect_ai.log import read_eval_log
    p = log_path(cell)
    return (read_eval_log(p).samples or []) if p else []


def load_fab_labels():
    """Two-judge fabrication labels {cell: [bool per sample in read order]}.

    Local labels (if any) are loaded first and the released file is applied on
    top, so a released cell always uses its published labels.
    """
    out = {}
    for path in LABEL_FILES:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out.update(json.load(f))
    if not out:
        raise FileNotFoundError("No fabrication labels found under data/ or logs/.")
    return out
