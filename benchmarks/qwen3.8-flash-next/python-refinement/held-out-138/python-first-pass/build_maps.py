"""Rebuild the two editable Beamer maps from the saved first-pass journals."""
import json
import re
from collections import Counter
from pathlib import Path

folder = Path(__file__).resolve().parent
snapshot = json.loads((folder / "snapshot.json").read_text())
styles = {
    "no_grid": ("closedColor", r"\cdot", "Closed, no grid"),
    "wrong": ("wrongColor", r"\times", "Wrong grid"),
    "exact": ("exactColor", r"\checkmark", "Exact"),
    "open": ("openColor", r"\triangleright", "Started, open"),
    "unattempted": ("white", "", "Not attempted"),
}

def draw_arm(arm):
    rows = sorted((row for row in snapshot["rows"] if row["arm"] == arm), key=lambda row: row["task_id"])
    ids = [row["task_id"] for row in rows]
    assert len(ids) == len(set(ids)) == 138 and all(re.fullmatch("[0-9a-f]{8}", task) for task in ids)
    counts = Counter(row["status"] for row in rows)
    closed = sum(row["done"] for row in rows)
    assert closed == counts["no_grid"] + counts["wrong"] + counts["exact"]
    assert sum(row["returned_grids"] > 0 for row in rows) == counts["wrong"] + counts["exact"]
    cells = []
    for row in rows:
        color, mark, _ = styles[row["status"]]
        cells.append(r"\pycell{%s}{%s}{%s}" % (color, row["task_id"], mark))
    lines = [r"{\centering\setlength{\tabcolsep}{0pt}\renewcommand{\arraystretch}{0.75}",
             r"\begin{tabular}{@{}*{10}{l}@{}}"]
    for start in range(0, len(cells), 10):
        lines.append(" & ".join(cells[start:start+10]) + r" \\[.5mm]")
    lines.extend([r"\end{tabular}\par}", r"\vspace{2mm}", r"\noindent\hbox to\linewidth{"])
    keys = []
    for state in ("unattempted", "open", "no_grid", "wrong", "exact"):
        color, mark, label = styles[state]
        keys.append(r"\pykey{%s}{%s}{%s}{%d}" % (color, mark, label, counts[state]))
    lines.append(r"\hfil".join(keys) + "}")
    (folder / f"map-{arm}.tex").write_text("\n".join(lines) + "\n")
    return ids

if __name__ == "__main__":
    assert draw_arm("A") == draw_arm("B")
