"""Export unnormalized Hessian block norms and spectra as CSV and SVG."""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import numpy as np


def color(t):
    palette = [(68,1,84), (59,82,139), (33,145,140), (94,201,98), (253,231,37)]
    t = max(0., min(1., t))*4
    i = min(int(t), 3); f = t-i
    return '#%02x%02x%02x' % tuple(round(a*(1-f)+b*f) for a,b in zip(palette[i], palette[i+1]))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('result_dir'); args = p.parse_args()
    root = Path(args.result_dir)
    data = json.loads((root/'frobenius.json').read_text(encoding='utf-8'))
    keys = data['keys']; norms = np.asarray(data['norm']); squared = np.asarray(data['norm_squared'])
    se = np.asarray(data['norm_squared_standard_error'])
    with (root/'frobenius_blocks.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['row','column','F_norm','F_norm_squared','squared_standard_error'])
        for i,row in enumerate(keys):
            for j,col in enumerate(keys): w.writerow([row,col,norms[i,j],squared[i,j],se[i,j]])
    pos = norms[norms > 0]
    if not pos.size: raise ValueError('all F-norm estimates are zero')
    lo, hi = math.log(float(pos.min())), math.log(float(pos.max())); span = max(hi-lo, 1e-30)
    cell = 14; left = top = 100; size = cell*len(keys)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size+270}" height="{size+190}" viewBox="0 0 {size+270} {size+190}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<text x="100" y="25" font-size="16">Absolute Hessian block Frobenius norms (log color scale)</text>']
    for i,key in enumerate(keys):
        name = html.escape(key); y=top+i*cell+cell*.75; x=left+i*cell+cell*.6
        s.append(f'<text x="{left-5}" y="{y}" font-size="8" text-anchor="end">{name}</text>')
        s.append(f'<text x="{x}" y="{top-5}" font-size="8" transform="rotate(-90 {x} {top-5})">{name}</text>')
        for j in range(len(keys)):
            value = norms[i,j]; t = (math.log(value)-lo)/span if value > 0 else 0
            s.append(f'<rect x="{left+j*cell}" y="{top+i*cell}" width="{cell}" height="{cell}" fill="{color(t)}"><title>{name} × {html.escape(keys[j])}: {value:.8g}</title></rect>')
    for i in range(7, len(keys), 7):
        x=left+i*cell; y=top+i*cell
        s.append(f'<path d="M{x} {top}v{size} M{left} {y}h{size}" stroke="white" stroke-width="0.7"/>')
    for i in range(100):
        y=top+i*2; s.append(f'<rect x="{left+size+25}" y="{y}" width="16" height="2" fill="{color(1-i/99)}"/>')
    s.extend([f'<text x="{left+size+47}" y="{top+9}" font-size="10">{math.exp(hi):.3g}</text>',
              f'<text x="{left+size+47}" y="{top+200}" font-size="10">{math.exp(lo):.3g}</text>',
              f'<text x="{left+size+25}" y="{top+220}" font-size="10">absolute F norm</text>', '</svg>'])
    (root/'frobenius_heatmap.svg').write_text('\n'.join(s), encoding='utf-8')
    folder = root/'spectra'; names = ['global'] + keys
    if all((folder/f'{name}.json').exists() for name in names):
        rows=[]; series=[]
        for name in names:
            info = json.loads((folder/f'{name}.json').read_text(encoding='utf-8'))
            arr = np.load(folder/f'{name}.npz'); values = arr['values'][:-1]; residuals=arr['relative_residual'][:-1]
            for rank,(value,residual) in enumerate(zip(values,residuals),1):
                rows.append([name,rank,value,residual,info['ordered_leading_space_certified'],info['boundary']['status']])
            if len(values): series.append((name, values))
        with (root/'spectra_top.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['scope','rank','eigenvalue','relative_residual','ordered_certified','boundary_status']); w.writerows(rows)
        ymin=min(float(v.min()) for _,v in series); ymax=max(float(v.max()) for _,v in series); width=800; height=500
        s=['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="600">','<rect width="100%" height="100%" fill="white"/>',
           '<text x="65" y="25" font-size="16">Global (black) and individual matrix block (blue) leading eigenvalues</text>',
           f'<path d="M65 50v{height}h{width}" fill="none" stroke="black"/>',
           f'<text x="5" y="60" font-size="11">{ymax:.3g}</text>',f'<text x="5" y="550" font-size="11">{ymin:.3g}</text>',
           '<text x="420" y="580" font-size="12">eigenvalue rank</text>']
        for name,values in reversed(series):
            pts=' '.join(f'{65+(i/max(len(values)-1,1))*width:.2f},{50+(ymax-float(v))/max(ymax-ymin,1e-30)*height:.2f}' for i,v in enumerate(values))
            s.append(f'<polyline points="{pts}" fill="none" stroke="{"black" if name=="global" else "steelblue"}" stroke-opacity="{1 if name=="global" else .25}" stroke-width="{2 if name=="global" else 1}"/>')
        s.append('</svg>'); (root/'spectra_distribution.svg').write_text('\n'.join(s),encoding='utf-8')


if __name__ == '__main__': main()
