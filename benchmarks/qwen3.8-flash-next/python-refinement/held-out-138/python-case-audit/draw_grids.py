"""Render the recorded arrays in figures.json. Requires ReportLab."""
import json
from itertools import groupby
from pathlib import Path
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

HERE = Path(__file__).resolve().parent
COLORS = ['#101010','#0074d9','#ff4136','#2ecc40','#ffdc00','#aaaaaa',
          '#f012be','#ff851b','#7fdbff','#870c25']

def plot(path, panels):
    panel_width, height = 178, 165
    pdf = Canvas(str(path), pagesize=(len(panels)*panel_width,height), invariant=True)
    # Shared scale preserves output size differences within each task.
    max_rows = max(len(p['grid']) for p in panels)
    max_cols = max(len(p['grid'][0]) for p in panels)
    cell = min(138 / max_rows, 158 / max_cols)
    for index, panel in enumerate(panels):
        data = panel['grid']
        x0 = panel_width*(index+.5) - cell*len(data[0])/2
        top = 143
        pdf.setFont('Helvetica',11.5)
        pdf.setFillColor(HexColor('#202732'))
        pdf.drawCentredString(panel_width*(index+.5),151,panel['label'])
        for r,row in enumerate(data):
            col = 0
            for value, run in groupby(row):
                length = sum(1 for _ in run)
                pdf.setFillColor(HexColor(COLORS[value]))
                pdf.rect(x0+cell*col,top-cell*(r+1),cell*length,cell,fill=1,stroke=0)
                col += length
        pdf.setStrokeColor(HexColor('#777777'))
        pdf.setLineWidth(.23)
        for r in range(len(data)+1):
            pdf.line(x0,top-cell*r,x0+cell*len(data[0]),top-cell*r)
        for c in range(len(data[0])+1):
            pdf.line(x0+cell*c,top,x0+cell*c,top-cell*len(data))
    pdf.save()

if __name__ == '__main__':
    for name, panels in json.loads((HERE/'figures.json').read_text()).items():
        plot(HERE/(name+'.pdf'),panels)
