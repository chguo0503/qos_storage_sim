#!/usr/bin/env python3
"""Mixed on every NPU, with explicit short startup and long homogeneous blocks."""
from concurrent.futures import ProcessPoolExecutor
import argparse,json,math
from block_issue_probe import simulate,HERE,profile

def work(task):
    na,nb,right=task;ca,_=profile(200,3302);cb,_=profile(32,2448)
    cycles=math.ceil(right/(8*(na*ca+nb*cb)))+1
    g0=[0]*3+[1]*15+([0]*na+[1]*nb)*cycles
    g1=[1]*15+[0]*3+([1]*nb+[0]*na)*cycles
    r=simulate([g0]*8+[g1]*8,right=right);r['name']=f'prefix3A15B_blocks{na}A{nb}B'
    r['construction']={'group0_prefix':'AAA'+'B'*15,'group1_prefix':'B'*15+'AAA',
                       'group0_cycle':'A'*na+'B'*nb,'group1_cycle':'B'*nb+'A'*na,
                       'main_cycles':cycles,'num_npu_per_group':8,
                       'counts_A_per_npu':3+cycles*na,'counts_B_per_npu':15+cycles*nb,
                       'requested_right_ms':right}
    return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--right-ms',type=float,default=60000.);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(work,[(na,nb,a.right_ms) for na,nb in [(10,40),(10,45),(10,50),(20,90)]]))
    rows.sort(key=lambda r:r['U'])
    (HERE/f'block_prefixed_role_swap_{int(a.right_ms)}ms.json').write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:print(r['name'],r['U'],r['windows'],r['all_warm_AB'],flush=True)
