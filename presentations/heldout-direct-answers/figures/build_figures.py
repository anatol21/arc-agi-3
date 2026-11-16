"""Rebuild the six slide figures using Python and ReportLab (originally 4.4.9).

    python build_figures.py
    python build_figures.py --refresh-from-workspace "/path/to/World Models"

The default uses only figure-inputs.json beside this file. The optional refresh
reconstructs those inputs from the original benchmark data and scoring audit.
"""
import argparse
import hashlib
import json
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

DEST = Path(__file__).resolve().parent
INPUT_PATH = DEST / "figure-inputs.json"
TRAIN_IDS = ("73251a56", "c3f564a4", "29ec7d0e")
PALETTE = {
    0: "#101010", 1: "#0074d9", 2: "#ff4136", 3: "#2ecc40", 4: "#ffdc00",
    5: "#aaaaaa", 6: "#f012be", 7: "#ff851b", 8: "#7fdbff", 9: "#870c25",
}
INK, BORDER = "#202732", "#505966"
ERR_PALETTE = {0: "#edf0f1", 1: "#0072b2", 2: "#d55e00"}
UNWRITTEN = "#e3e5e6"


def refresh(root):
    archive = root / "artifacts/cluster/arc-agi3-data-v1"
    manifest = json.loads((archive / "manifest.json").read_text())
    hashes = {}

    def indexed(name):
        payload = (archive / name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        assert digest == manifest["files_sha256"][name], name
        hashes[name] = digest
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        result = {row["task_id"]: row for row in rows}
        assert len(result) == len(rows)
        return result

    tasks = indexed("evaluation/tasks.jsonl")
    answers = indexed("scoring/answers.jsonl")
    train = indexed("data/train.jsonl")
    audit_bytes = (root / "research/evidence/direct-answer-audit/audit.json").read_bytes()
    hashes["direct-answer-audit/audit.json"] = hashlib.sha256(audit_bytes).hexdigest()
    audit = json.loads(audit_bytes)
    primary = [r for r in audit["observations"] if r["precedence"] == 0]
    exact = {}
    for arm in ("base", "finetuned"):
        arm_rows = [r for r in primary if r["condition"] == arm]
        assert len(arm_rows) == len({r["task_id"] for r in arm_rows}) == 138
        exact[arm] = sorted(r["task_id"] for r in arm_rows if r["task_correct"])

    public = {}
    for task in sorted(set(exact["finetuned"]) | set(exact["base"])):
        queries = tasks[task]["test"]
        targets = answers[task]["outputs"]
        assert len(queries) == len(targets) == 1
        public[task] = {"input": queries[0]["input"], "output": targets[0]}

    training = {}
    for task in TRAIN_IDS:
        assert task in manifest["task_ids"]["train"] and len(train[task]["test"]) == 1
        training[task] = train[task]["test"][0]

    focus = {}
    for task in ("e66aafb8", "e95e3d8e", "1e97544e", "af22c60d"):
        arms = ["base", "finetuned"]
        if task == "af22c60d":
            arms += ["low_nucleus", "instruct_sampling"]
        focus[task] = {arm: audit["focus"][task][arm] for arm in arms}

    subset = {"source_sha256": hashes, "public_queries": public, "training_queries": training,
              "primary_exact": exact, "focus": focus}
    INPUT_PATH.write_text(json.dumps(subset, separators=(",", ":"), sort_keys=True) + "\n")


def query(task, training=False):
    return INPUTS["training_queries" if training else "public_queries"][task]


def fig(name, width, height):
    return canvas.Canvas(str(DEST / f"{name}.pdf"), pagesize=(width, height), pageCompression=1, invariant=True)


def label(page, x, y, text, *, size=13, mono=False, color=INK, align="center"):
    page.setFont("Courier-Bold" if mono else "Helvetica", size)
    page.setFillColor(HexColor(color))
    if align == "center":
        page.drawCentredString(x, y, text)
    else:
        page.drawString(x, y, text)


def grid(page, data, cx, top, cell, *, colors=PALETTE, digits=False, outlines=True):
    """Draw in original row-major orientation; coordinate arguments are PDF points."""
    rows, cols = len(data), len(data[0])
    assert all(len(row) == cols for row in data)
    left = cx - cell * cols / 2
    # Run-length rectangles avoid seams between adjacent cells of the same color.
    for r, row in enumerate(data):
        c = 0
        while c < cols:
            end = c + 1
            while end < cols and row[end] == row[c]:
                end += 1
            page.setFillColor(HexColor(colors[row[c]]))
            page.rect(left + c * cell, top - (r + 1) * cell, (end-c) * cell, cell, fill=1, stroke=0)
            c = end
    if outlines:
        page.saveState()
        page.setStrokeGray(0.2)
        page.setStrokeAlpha(0.28)
        page.setLineWidth(min(.42, cell * .05))
        for c in range(1, cols):
            page.line(left+c*cell, top, left+c*cell, top-rows*cell)
        for r in range(1, rows):
            page.line(left, top-r*cell, left+cols*cell, top-r*cell)
        page.restoreState()
    page.setLineWidth(.65)
    page.setStrokeColor(HexColor(BORDER))
    page.rect(left, top-cell*rows, cols*cell, rows*cell)
    if digits:
        for r,row in enumerate(data):
            for c,v in enumerate(row):
                col = PALETTE[v].lstrip('#')
                lum = sum(w * int(col[i:i+2],16) for i,w in ((0,.2126),(2,.7152),(4,.0722)))
                label(page, left+(c+.5)*cell, top-(r+.68)*cell, str(v), size=cell*.57, mono=True, color="#ffffff" if lum<110 else "#14191f")
    return left, top


def pairs_figure(name, ids, *, training=False):
    width = 696
    height = 272
    page = fig(name, width, height)
    page.setTitle(name)
    margin_left = 58
    column_width = (width-margin_left)/len(ids)
    side=102
    for col,t in enumerate(ids):
        x=margin_left+(col+.5)*column_width
        q=query(t, training)
        label(page, x, 254, t, mono=True, size=15)
        for key, y in (("input",239),("output",113)):
            data=q[key]
            grid(page, data, x, y, side/max(map(len,data),default=1))
    for y, text in ((184,"Input"), (57,"Output")):
        label(page, 4, y, text, align="left", size=13)
    page.save()


def neighbors():
    ids = TRAIN_IDS
    page = fig("training-neighbors", 690, 144)
    for i, task in enumerate(ids):
        q = query(task, training=True)
        center = 115 + 230*i
        label(page, center, 127, task, mono=True, size=15)
        for dx, key, caption in [(-55, 'input', 'Input'), (55, 'output', 'Target')]:
            data = q[key]
            grid(page, data, center+dx, 113, 94/len(data))
            label(page, center+dx, 3, caption, size=11)
    page.save()


def e66():
    t="e66aafb8"
    d=INPUTS["focus"][t]
    q=query(t); inp=q["input"]; out=q["output"]
    base=d['base']['prediction']
    assert d['finetuned']['prediction'] == out == base[:3]
    assert base[3:] == [out[1],out[0]]
    assert {(r,c) for r,row in enumerate(inp) for c,v in enumerate(row) if not v}=={(r,c) for r in range(9,12) for c in range(14,17)}
    page=fig(t,600,174)
    label(page,87,156,"Query input")
    left,top=grid(page,inp,87,144,5.75)
    page.setStrokeColor(HexColor("#ffffff"))
    page.setLineWidth(1.6)
    page.rect(left+14*5.75,top-12*5.75,3*5.75,3*5.75)
    for x,title,g in ((287,"NF4 base",base),(500,"Checkpoint-660",out)):
        label(page,x,156,title)
        grid(page,g,x,139,25,digits=True)
    page.setLineWidth(1.9)
    page.setStrokeColor(HexColor("#a52131"))
    page.rect(287-37.5,139-125,75,50)
    page.save()


def e95():
    t='e95e3d8e'; q=query(t); d=INPUTS['focus'][t]
    assert d['base']['missing_outer_list'] and not d['base']['errors'] and not d['finetuned']['errors']
    assert sum(row.count(0) for row in q['input'])==46
    page=fig(t,524,174)
    for x,title,key in ((147,'Query input','input'),(375,'Cells returned by both models','output')):
        label(page,x,156,title,size=13)
        grid(page,q[key],x,143,6.27)
    page.save()


def repeated():
    task='1e97544e'; q=query(task); d=INPUTS['focus'][task]
    assert not d['finetuned']['errors'] and d['base']['wrong_on_blank']==22 and d['base']['wrong_on_visible']==13
    errors=[[0]*23 for _ in range(23)]
    for r,c,initial,pred,correct in d['base']['errors']:
        assert q['input'][r-1][c-1]==initial and q['output'][r-1][c-1]==correct and pred==1+correct%9
        errors[r-1][c-1]=1 if initial==0 else 2
    assert d['base']['errors'][0]==[5,5,0,7,6]
    page=fig(task,612,195)
    for x,lab,matrix,col in ((95,'Query input',q['input'],PALETTE),(306,'Checkpoint-660',q['output'],PALETTE),(515,'NF4 base errors',errors,ERR_PALETTE)):
        label(page,x,180,lab,size=14)
        grid(page,matrix,x,168,6.45,colors=col)
    x=93
    for index,lab in ((1,"Incorrect fill of a blank cell (22)"),(2,"Changed a visible cell (13)")):
        page.setFillColor(HexColor(ERR_PALETTE[index])); page.rect(x,3,9,9,fill=1,stroke=0)
        label(page,x+15,3,lab,size=11,align='left'); x+=232
    page.save()


def partial():
    task='af22c60d'
    q=query(task); d=INPUTS['focus'][task]
    assert not d['base']['errors'] and not d['finetuned']['errors']
    for arm in ('low_nucleus','instruct_sampling'):
        s=d[arm]
        assert s['numeric_values']==684 and s['completed_row_lengths']==[30]*22 and s['prefix_errors']==0
    expected=q['output']
    observed=[[v if r*30+c<684 else -1 for c,v in enumerate(row)] for r,row in enumerate(expected)]
    page=fig(task,609,178)
    for x,lab,matrix in ((96,'Query input',q['input']),(305,'NF4 (both models)',expected),(514,'BF16 (both settings)',observed)):
        label(page,x,161,lab)
        grid(page,matrix,x,149,4.85,colors={**PALETTE,-1:UNWRITTEN})
    page.save()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-from-workspace", type=Path, metavar="ROOT")
    args = parser.parse_args()
    if args.refresh_from_workspace is not None:
        refresh(args.refresh_from_workspace)
    INPUTS = json.loads(INPUT_PATH.read_text())
    exact = {arm: set(ids) for arm, ids in INPUTS["primary_exact"].items()}
    shared = sorted(exact['base'] & exact['finetuned'])
    tuned = sorted(exact['finetuned'] - exact['base'])
    assert shared==['4aab4007','af22c60d','c663677b','ca8f78db']
    assert tuned==['1e97544e','e66aafb8','e95e3d8e'] and not exact['base']-exact['finetuned']
    pairs_figure("nf4-shared",shared)
    neighbors()
    e66()
    e95()
    repeated()
    partial()
