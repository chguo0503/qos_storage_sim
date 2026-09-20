"""APPROXIMATE FIFO candidate selector; never a substitute for native simulation.

One disk, 40 GiB/s; 8 serial NPUs; 8 layers. Each whole layer is enqueued
atomically, unlike native per-IO submission. Reads pipeline with computation;
next-request L0 is issued at previous-request final-layer compute start.
Data interpolation is confined to the measured grid. All lanes independently
shuffle the same type counts and draw unique (total length, NQL) profiles.
"""
from __future__ import annotations
import argparse, ast, bisect, functools, heapq, itertools, json, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLE = ast.literal_eval((ROOT / 'data').read_text())
XS = sorted({k[0] for k in TABLE})
YS = sorted({k[1] for k in TABLE})
LAYERS = 8

def bracket(grid, value):
    if not grid[0] <= value <= grid[-1]:
        raise ValueError('No extrapolation')
    j = bisect.bisect_left(grid, value)
    if grid[j] == value:
        return value, value, 0.
    a, b = grid[j-1:j+1]
    return a, b, (value-a)/(b-a)

@functools.lru_cache(None)
def profile(k, nql):
    x0,x1,wx = bracket(XS,k)
    y0,y1,wy = bracket(YS,nql)
    c = sum(TABLE[x,y][1]*vx*vy for x,vx in ((x0,1-wx),(x1,wx))
            for y,vy in ((y0,1-wy),(y1,wy))) / 1000
    v = (int(k*1024)-nql)*1408/2**30
    return c, v

def decks(specs, seed=7, horizon=4500.):
    nominal = sum(w*profile(k,n)[0] for k,n,w in specs)*LAYERS
    cycles = math.ceil(horizon/nominal)+1
    counts = [cycles*w for k,n,w in specs]
    choices = []
    for (k,n,w),count in zip(specs,counts):
        size = max(count+8,math.ceil(1.25*count))
        if n+size-1 <= 4096:
            choices.append(range(n,n+size))
        else:
            choices.append(range(n-size+1,n+1))
        if min(choices[-1]) < 64:
            raise ValueError('Too many distinct profiles')
    pattern = [i for i,w in enumerate(counts) for _ in range(w)]
    lanes = []
    for lane in range(8):
        rng = random.Random(seed+100003*lane)
        roles = list(pattern)
        rng.shuffle(roles)
        sampled = [iter(rng.sample(list(c),count)) for c,count in zip(choices,counts)]
        seen = set()
        deck = []
        for role in roles:
            k,n,w = specs[role]
            q = next(sampled[role])
            if (k,q) in seen:
                raise ValueError('Overlapping type pools')
            seen.add((k,q))
            c,v = profile(k,q)
            deck.append((role,c,v,k,q))
        lanes.append(deck)
    return lanes

def native_matched_decks(spec, seed=7, horizon_ms=4500.):
    """Match run_random_multitype.build exactly, omitting block placement only.

    This was checked against full native build request sequences. Do not use
    the old one-sided `decks` generator for native matched robustness tests.
    """
    groups=spec['groups'];ssu=spec.get('ssu',1)
    cycle_ms=LAYERS*sum(g['weight']*profile(g['total_k'],g['nql'])[0] for g in groups)
    cycles=math.ceil(horizon_ms/cycle_ms)+1
    while True:
        pools={};used=set()
        for g in groups:
            count=cycles*g['weight'];size=math.ceil(count*1.15)+4
            candidates=sorted(range(64,4097),key=lambda y:(abs(y-g['nql']),y))
            ys=[]
            for y in candidates:
                key=(int(round(g['total_k']*1024)),y)
                c,v=profile(g['total_k'],y)
                if key not in used and v/(c/1000)<=min(40*ssu,50)*(1-1e-9):
                    ys.append(y);used.add(key)
                if len(ys)==size:break
            if len(ys)<size:raise ValueError('Not enough unique profiles')
            pools[g['id']]=ys
        lower_ms=LAYERS*sum(cycles*g['weight']*min(profile(g['total_k'],y)[0] for y in pools[g['id']]) for g in groups)
        if lower_ms>horizon_ms:break
        cycles+=1
    lanes=[];gmap={g['id']:(i,g) for i,g in enumerate(groups)}
    for n in range(8):
        rng=random.Random(seed+100003*n)
        deck=[g['id'] for g in groups for _ in range(cycles*g['weight'])]
        rng.shuffle(deck)
        picks={g['id']:iter(rng.sample(pools[g['id']],cycles*g['weight'])) for g in groups}
        lane=[]
        for gid in deck:
            i,g=gmap[gid];q=next(picks[gid]);c,v=profile(g['total_k'],q)
            lane.append((i,c,v,g['total_k'],q))
        lanes.append(lane)
    return lanes

def simulate(specs, seed=7, left=2000., right=4000., n_ssu=1, matched_spec=None,
             actual_ring_vectors=None, enforce_npu_link=False):
    # Multi-SSU surrogate assumes perfectly balanced layer bytes; true ring
    # placement must be measured by the native model.
    lanes = decks(specs,seed) if matched_spec is None else native_matched_decks(matched_spec,seed)
    disk_free = [0.]*n_ssu
    events = []
    serial = itertools.count()
    busy = [0.]*8
    bytype = [{'compute_ms':0.,'stall_ms':0.,'l0_stall_ms':0.,'internal_stall_ms':0.} for _ in specs]
    nseen = [set() for _ in range(8)]
    cap = 40.*n_ssu
    def clip(a,b):
        return max(0.,min(b,right)-max(a,left))
    # Tiny last block link tail approximates pipelined 50 GiB/s NPU receipt.
    def read_at(release, volume, lane_id, request_index):
        vector=actual_ring_vectors[lane_id,request_index] if actual_ring_vectors is not None else [volume/n_ssu]*n_ssu
        edges=[]
        for s,v in enumerate(vector):
            begin=max(disk_free[s],release)
            end=begin+v/40*1000
            disk_free[s]=end
            edges.extend(((begin,40.),(end,-40.)))
        edges.sort()
        if enforce_npu_link:
            last=edges[0][0];backlog=0.;arrival_rate=0.
            for now,delta in edges:
                backlog=max(0.,backlog+(arrival_rate-50)*(now-last)/1000)
                arrival_rate+=delta;last=now
            ready=last+backlog/50*1000
        else:
            ready=max(disk_free)
        return ready + 176/1024**2/50*1000
    for n,deck in enumerate(lanes):
        ready = read_at(0.,deck[0][2],n,0)
        heapq.heappush(events,(ready,next(serial),n,0,0,0.))
    while events:
        start,_,n,r,l,barrier = heapq.heappop(events)
        role,c,v,k,q = lanes[n][r]
        stall = clip(barrier,start)
        bytype[role]['stall_ms'] += stall
        bytype[role]['l0_stall_ms' if l == 0 else 'internal_stall_ms'] += stall
        if start >= right:
            continue
        end = start+c
        got = clip(start,end)
        busy[n] += got
        bytype[role]['compute_ms'] += got
        if got:
            nseen[n].add(role)
        nr,nl = (r,l+1) if l<7 else (r+1,0)
        if nr < len(lanes[n]):
            ready = read_at(start,lanes[n][nr][2],n,nr)
            heapq.heappush(events,(max(end,ready),next(serial),n,nr,nl,end))
    for t in bytype:
        t['utilization_pct'] = 100*t['compute_ms']/(t['compute_ms']+t['stall_ms']) if t['compute_ms'] else None
    vol = sum(p[2] for lane in lanes for p in lane)
    comp = sum(p[1] for lane in lanes for p in lane)
    upper = sum(max(p[2]/p[1]*1000 for p in lane) for lane in lanes)
    native_mean = sum(sum(p[2] for p in lane)/sum(p[1] for p in lane)*1000 for lane in lanes)
    return dict(approximate_only=True,seed=seed,num_ssu=n_ssu,
        profiles=[dict(total_k=k,nql=n,weight=w,compute_ms=profile(k,n)[0],
                       read_gib=profile(k,n)[1],B_gib_s=profile(k,n)[1]/profile(k,n)[0]*1000,
                       data_direct=(k,n) in TABLE) for k,n,w in specs],
        utilization_pct=sum(busy)/(8*(right-left))*100,
        lane_utilization_pct=[x/(right-left)*100 for x in busy],bytype=bytype,
        input_mean_demand_gib_s=8*vol/comp*1000,
        input_mean_load=8*vol/comp*1000/cap,
        native_definition_mean_load=native_mean/cap,
        nominal_max_current_load=upper/cap,
        all_lanes_all_roles_warm=all(len(s)==len(specs) for s in nseen),
        warm_group_indices_by_npu=[sorted(s) for s in nseen],
        counts_per_npu=[sum(p[0]==i for p in lanes[0]) for i in range(len(specs))])

def search(seed=7, draws=4000):
    rng = random.Random(73419)
    candidates = []
    seen = set()
    def consider(specs,group):
        specs=tuple(sorted(specs))
        if specs in seen:
            return
        seen.add(specs)
        base_v = sum(w*profile(k,n)[1] for k,n,w in specs)
        base_c = sum(w*profile(k,n)[0] for k,n,w in specs)
        nominal = 8*base_v/base_c*1000/40
        if not .25 < nominal < .965:
            return
        try:
            row = simulate(specs,seed)
        except ValueError:
            return
        if row['input_mean_load'] < .95:
            row['search_group']=group
            candidates.append(row)
    for lk in (128,200):
        for ln in (1536,2048,2560,3072,3584,4096):
            for sn in (128,256,384,512,640,768,1024,1280,1536,2048):
                for sw in (1,2,3,4,6,8,12,16,24,32):
                    consider([(lk,ln,1),(32,sn,sw)],'two_type_grid')
    pool = [(k,n) for k in (32,48,64,80,96,128,160,200)
            for n in (128,256,384,512,640,768,1024,1280,1536,1792,2048,2560,3072,3584,4096)]
    for draw in range(draws):
        count = rng.choice((3,3,4,4,5,6))
        if draw%2:
            selected=rng.sample(pool,count)
        else:
            selected=rng.sample([p for p in pool if p[0] in (32,128)],count)
        weights=[rng.choice((1,1,2,3,4,6,8,12)) for _ in selected]
        consider([(k,n,w) for (k,n),w in zip(selected,weights)],'multitype_random')
    candidates.sort(key=lambda r:r['utilization_pct'])
    return dict(description='APPROXIMATE screening only: whole-layer FIFO, perfect SSU balance, no native QoS or per-IO client events.',
                search_seed=73419,workload_seed=seed,evaluated=len(candidates),
                assumptions=['8 NPUs each mixed, independent random order','8 layers, batch 1',
                             'prequeued requests; next L0 at previous last compute start',
                             'exact KV prefix bytes; interpolation inside measured data grid',
                             'window [2000,4000) ms; input ideal horizon >4500ms'],
                rows=candidates)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--draws',type=int,default=4000)
    p.add_argument('--out',type=Path,default=ROOT/'results/random_multitype_search_20260914/approx_candidates.json')
    a=p.parse_args()
    payload=search(draws=a.draws)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(payload,indent=2))
    print(json.dumps({'path':str(a.out),'evaluated':payload['evaluated'],'best':payload['rows'][:8]},indent=2))

if __name__=='__main__':
    main()
