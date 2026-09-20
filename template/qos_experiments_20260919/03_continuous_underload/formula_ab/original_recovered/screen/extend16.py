exec(open('/workspace/scratch/351ecd930597/xy_experiment_screen/screen.py').read().split('rs=[]')[0])
from run_fixed128_32 import bracket
import math

def ext(k,q):
 if k>=20:return profile(table,int(k*1024)-q,q)
 x0,x1,wx=bracket(sorted({x for x,y in table}),k,allow_low=True)
 y0,y1,wy=bracket(sorted({y for x,y in table}),q)
 anchors=[]
 for x,vx in ((x0,1-wx),(x1,wx)):
  for y,vy in ((y0,1-wy),(y1,wy)):
   if vx*vy:anchors.append(dict(seq_len_k=x,nql=y,weight=vx*vy,compute_us=table[x,y][1]))
 c=sum(a['weight']*a['compute_us'] for a in anchors);v=(k*1024-q)*1408/2**30
 return dict(total_tokens=k*1024,total_length_k=k,nql=q,ssd_prefix_tokens=k*1024-q,compute_us=c,read_gib=v,B_gib_s=v/(c/1e6),constructed_profile=True,profile_construction=dict(method='bilinear_length_extrapolation_nql_interpolation',source='data',anchors=anchors,extrapolated=True,compute_scale=1,kv_formula='exact hit prefix tokens * 1408 / 2**30; partial final block'))
if __name__=='__main__':
 out=[]
 for kA in [128,144,160,176,192,200]:
  for qA in [512,1024,1536,2048]:
   for kB in [16,18,20,24]:
    for qB in [512,1024,1536,2048]:
     a=ext(kA,qA);b=ext(kB,qB);x=a['read_gib']/b['read_gib'];y=a['compute_us']/b['compute_us']
     if not (x>8 and 1<y<8):continue
     ba=a['B_gib_s']*2**30/1e9;bb=b['B_gib_s']*2**30/1e9
     for na in range(1,8):
      for s in range(1,5):
       d=na*ba+(8-na)*bb;block=a['read_gib']*2**30/1e9/(40*s)/(b['compute_us']/1e6)
       if d/(40*s)<.98 and block>1:out.append((kA,qA,kB,qB,na,s,x,y,d/(40*s),block))
 for r in sorted(out,key=lambda r:r[-1],reverse=True):print(r)
