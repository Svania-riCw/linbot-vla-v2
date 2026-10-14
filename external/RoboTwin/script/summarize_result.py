"""Append clean / random / overall averages to a grouped-run _result.txt (in place, idempotent).

Input lines are what eval_policy.py appends per (task, case) unit:
    <task>\t<case>\t<suc>/<total>\t<rate>
Output: the unit lines sorted by (case, task), followed by a '#'-prefixed summary block that is
dropped and recomputed on every call, so the script can be re-run on a partial or finished run:
    # all_clean_avg   0.4200  (7 units, 73/175 episodes)
    # all_random_avg  ...
    # all_avg         ...     (14 units, ...)
Averages are the mean of per-unit success rates (each task weighs the same even if trial
counts differ); the pooled episode counts are shown alongside.

Usage: python script/summarize_result.py eval_result/<model>/<run>/_result.txt
"""

import sys
from pathlib import Path

CASES = ("clean", "random")


def parse(lines):
    units = {}
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or "/" not in parts[2]:
            continue
        suc, tot = (int(x) for x in parts[2].split("/"))
        units[(parts[0], parts[1])] = (suc, tot)   # a re-run unit overwrites its earlier line
    return units


def summary_line(name, rows):
    if not rows:
        return f"# {name}\tn/a\t(0 units)"
    mean = sum(s / t for s, t in rows) / len(rows)
    suc, tot = sum(s for s, _ in rows), sum(t for _, t in rows)
    return f"# {name}\t{mean:.4f}\t({len(rows)} units, {suc}/{tot} episodes)"


def summarize(path):
    path = Path(path)
    units = parse(path.read_text().splitlines())
    order = sorted(units, key=lambda k: (CASES.index(k[1]) if k[1] in CASES else len(CASES), k[1], k[0]))
    out = [f"{task}\t{case}\t{units[(task, case)][0]}/{units[(task, case)][1]}\t"
           f"{units[(task, case)][0] / units[(task, case)][1]}" for task, case in order]
    out.append("# ---- summary: mean of per-unit success rates ----")
    for case in CASES:
        out.append(summary_line(f"all_{case}_avg", [v for (t, c), v in units.items() if c == case]))
    out.append(summary_line("all_avg", list(units.values())))
    path.write_text("\n".join(out) + "\n")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print("\n".join(summarize(sys.argv[1])))
