"""Export per-condition scientific plots and tables from seven_measure results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows, fieldnames):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', help='one or more seven_measure output directories')
    parser.add_argument('--output', required=True, help='new output directory')
    args = parser.parse_args()
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('Plotting needs matplotlib: pip install -e ".[plots]"') from exc
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'figure.dpi':150, 'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.grid':True, 'grid.alpha':.15})
    summary=[]; notes=['# Seven-block positive-subspace report', '',
                       'Each condition is displayed separately; seeds, batches and checkpoints are not pooled.', '']
    for index, name in enumerate(args.inputs, 1):
        src = Path(name); meta = read(src/'metadata.json'); coverage = read(src/'coverage.json')
        status = read(src/'status.json'); ritz = read(src/'ritz.json'); norms = read(src/'frobenius.json')
        condition = f'condition_{index:02d}'; folder = output/condition; folder.mkdir()
        notes += [f'## {condition}', '', f'- Input: `{src.resolve()}`',
                  f"- Step {meta['step']}; {meta['scope']}; valid tokens {meta['valid_tokens']}; status `{status['phase']}`.",
                  f"- Checkpoint SHA256 `{meta['checkpoint_sha256']}`; batch SHA256 `{meta['batch_sha256']}`.", '']
        preliminary=sum(row['status'] != 'complete' for row in coverage)
        if preliminary:
            notes += [f'- **{preliminary} coverage rows are preliminary or unavailable:** inspect status and spectral-boundary details in coverage.json.', '']
        for filename,label in [('equal_dimension_control.json','Equal-dimension S4 control'),
                               ('diagonal_equal_dimension_control.json','Equal-dimension D7 control'),
                               ('cross_randomization.json','Synthetic cross randomization')]:
            if (src/filename).exists():
                optional=read(src/filename)
                notes += [f"- {label}: `{optional.get('status', 'unknown')}`; see `{filename}` in the input directory.", '']
        for row in coverage:
            g=row.get('geometry', {})
            summary.append(dict(condition=condition, step=meta['step'], scope=meta['scope'],
                training_seed=(meta.get('training_arguments') or {}).get('seed'),
                comparison=row['comparison'], target_rank=row['target_rank'], self_rank=row['self_rank'],
                status=row['status'], candidate_dim=g.get('candidate_dim'),
                containment=g.get('H_to_candidate_containment'), overlap=g.get('overlap'),
                minimum_angle=(g.get('angle_range_deg') or [None,None])[0],
                maximum_angle=(g.get('angle_range_deg') or [None,None])[1],
                random_expected=row.get('random_block_baseline',{}).get('expected')))
        usable=[row for row in coverage if row.get('geometry') and row['comparison'] in ('S4_H7','S7_H7')]
        if usable:
            ranks=sorted(set(row['target_rank'] for row in usable))
            fig, axes=plt.subplots(1,len(ranks),figsize=(6*len(ranks),4.5),squeeze=False,layout='constrained')
            for ax,rank in zip(axes[0],ranks):
                for comparison,color in [('S4_H7','#197e98'),('S7_H7','#d36538')]:
                    rows=sorted((x for x in usable if x['target_rank']==rank and x['comparison']==comparison),key=lambda x:x['self_rank'])
                    if rows:
                        ax.plot([x['self_rank'] for x in rows],[x['geometry']['H_to_candidate_containment'] for x in rows],
                                'o-',label=comparison,color=color)
                        ax.plot([x['self_rank'] for x in rows],[x['random_block_baseline']['expected'] for x in rows],
                                '--',color=color,alpha=.6,label=comparison+' random expectation')
                ax.set(title=f'H7 positive top {rank}',xlabel='Positive self directions per block (requested)',ylabel='H to candidate containment',ylim=(0,1.02))
                ax.legend(fontsize=8)
            fig.suptitle('Inspect coverage_summary.csv for preliminary or unavailable boundaries', fontsize=9)
            fig.savefig(folder/'coverage.png');plt.close(fig)
            maxrank=max(ranks); m=max(x['self_rank'] for x in usable if x['target_rank']==maxrank)
            chosen=[x for x in usable if x['target_rank']==maxrank and x['self_rank']==m]
            fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
            for row in chosen:
                g=row['geometry'];name=row['comparison']
                axes[0].plot(np.arange(1,len(g['capture_each_mode'])+1),g['capture_each_mode'],'o-',ms=3,label=name)
                angles=g['principal_angles_deg']
                axes[1].plot(np.arange(1,len(angles)+1),angles,'o-',ms=3,label=name)
            axes[0].set(title=f'Per-mode capture: H7 top {maxrank}',xlabel='H mode',ylabel='capture',ylim=(0,1.02))
            axes[1].set(title='Full principal-angle sequence',xlabel='Angle index',ylabel='degrees',ylim=(0,90))
            for ax in axes:ax.legend()
            fig.savefig(folder/'capture_angles.png');plt.close(fig)
            notes += [f'![Coverage]({condition}/coverage.png)', '',
                      f'![Mode capture and angles]({condition}/capture_angles.png)', '']
        if 'blocks' in norms:
            order=norms['block_order'];fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
            blockrows=[]
            for ax,label in zip(axes,('H','A','R')):
                matrix=np.zeros((len(order),len(order)))
                for pair,record in norms['blocks'].items():
                    i,j=order.index(pair[0]),order.index(pair[1]);matrix[i,j]=matrix[j,i]=record[label]['norm']
                    blockrows.append(dict(operator=label,block=pair,norm=record[label]['norm'],
                        norm2=record[label]['norm2'],norm2_stderr=record[label]['norm2_stderr'],
                        norm2_ci95_low=record[label]['norm2_ci95'][0],norm2_ci95_high=record[label]['norm2_ci95'][1],
                        element_rms=record[label]['element_rms']))
                im=ax.imshow(matrix,cmap='viridis');ax.set(xticks=range(len(order)),yticks=range(len(order)),
                    xticklabels=order,yticklabels=order,title=label+' full-block Frobenius norm');ax.grid(False)
                fig.colorbar(im,ax=ax,shrink=.75)
            fig.savefig(folder/'frobenius.png');plt.close(fig)
            write_csv(folder/'frobenius_blocks.csv',blockrows,['operator','block','norm','norm2','norm2_stderr','norm2_ci95_low','norm2_ci95_high','element_rms'])
            notes += [f'![Full-matrix Frobenius norms]({condition}/frobenius.png)', '',
                      'Frobenius norms cover the full matrices and all spectral signs; each panel has its own color scale.', '']
        fig,ax=plt.subplots(figsize=(6,4.5),layout='constrained'); plotted=False
        for key,color in [('H4','#197e98'),('S4_H7','#8f5ba5'),('H7','#d36538')]:
            row=ritz.get(key,{})
            if 'Ritz_values' not in row:continue
            x=np.arange(1,len(row['Ritz_values'])+1)
            y=[np.nan if value is None else value for value in row['Ritz_values']]
            if key != 'S4_H7':
                ax.plot(x,row['target_eigenvalues'],'o-',ms=3,color=color,label=key+' target')
            ax.plot(x,y,'s--',ms=3,color=color,label=key+' Ritz');plotted=True
        if plotted:
            ax.set(title='Positive endpoint: target versus Ritz',xlabel='rank',ylabel='eigenvalue');ax.legend(fontsize=8)
            fig.savefig(folder/'ritz.png');notes += [f'![Ritz spectra]({condition}/ritz.png)', '',
                'Ritz values are restricted-space curvatures; inspect full-space residuals in ritz.json.', '']
        plt.close(fig)
    fields=['condition','step','scope','training_seed','comparison','target_rank','self_rank','status',
            'candidate_dim','containment','overlap','minimum_angle','maximum_angle','random_expected']
    write_csv(output/'coverage_summary.csv',summary,fields)
    (output/'report.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    print(f'Wrote report to {output.resolve()}')


if __name__ == '__main__':
    main()
