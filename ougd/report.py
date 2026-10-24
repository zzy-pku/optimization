"""Export protocol tables and a self-contained SVG block-strength heatmap."""
import argparse
import json
from pathlib import Path
import math

def export(directory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = Path(directory)
    meta = json.loads((p/'operator_meta.json').read_text())
    common = {k:meta.get(k) for k in ('step','pre_update_state','layer','scope','operator','anchor','precision','valid_tokens','batch_sha256','checkpoint_sha256')}
    def table(name,rows):
        if rows: pq.write_table(pa.Table.from_pylist([{**common,**r} for r in rows]),p/f'{name}.parquet')
    norms = json.loads((p/'norms_and_coupling.json').read_text())
    table('norms_and_coupling',[dict(block=k,**v) for k,v in norms['blocks'].items()])
    table('coverage',json.loads((p/'coverage.json').read_text()))
    eig,svd = [],[]
    for f in p.glob('*.json'):
        if f.name.startswith(('H_','self_only_','drop_')):
            s = json.loads(f.read_text())
            for i,v in enumerate(s['values']):
                if f.stem.endswith('SVD'):
                    svd.append(dict(series=f.stem,index=i,value=v,converged=s['converged'],
                        left_residual=s['left_residual'][i],right_residual=s['right_residual'][i],
                        left_relative_residual=s['left_relative_residual'][i],right_relative_residual=s['right_relative_residual'][i]))
                else:
                    eig.append(dict(series=f.stem,index=i,value=v,converged=s['converged'],
                        residual=s['residual'][i],relative_residual=s['relative_residual'][i]))
    table('eigenvalues',eig)
    table('singular_values',svd)
    f = p/'contributions.json'
    if f.exists():
        rows = []
        for key,ends in json.loads(f.read_text()).items():
            if not isinstance(ends,dict): continue
            for end,records in ends.items():
                if end not in ('LA','SA') or not isinstance(records,list): continue
                for index,record in enumerate(records):
                    for block,value in record['signed_terms'].items():
                        rows.append(dict(parameter_set=key,eig_end=end,index=index,block=block,
                            signed_contribution=value,eigenvalue=record['eigenvalue'],sum_error=record['sum_error']))
        table('contributions',rows)
    f = p/'cross_subspaces.json'
    if f.exists():
        table('subspace_metrics',[dict(block=r['block'],metrics_json=json.dumps(r)) for r in json.loads(f.read_text())])
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="620" height="600">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="30" y="30">M1 H block Frobenius norm — {meta["scope"]}, step {meta["step"]}</text>']
    blocks = 'OUGD'
    values = [x['norm'] for x in norms['blocks'].values()]
    maxval = max(values) or 1
    for a,i in enumerate(blocks):
        svg.append(f'<text x="{145+110*a}" y="75">{i}</text>')
        svg.append(f'<text x="55" y="{140+110*a}">{i}</text>')
        for b,j in enumerate(blocks):
            key = i+j if a <= b else j+i
            val = norms['blocks'][key]['norm']
            shade = int(255-180*math.sqrt(val/maxval))
            svg.append(f'<rect x="{100+110*b}" y="{90+110*a}" width="105" height="105" fill="rgb(255,{shade},{shade})"/>')
            svg.append(f'<text x="{110+110*b}" y="{145+110*a}" font-size="13">{val:.4g}</text>')
    svg.append('</svg>')
    (p/'norm_heatmap.svg').write_text('\n'.join(svg))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory')
    export(p.parse_args().directory)

if __name__=='__main__': main()
