"""Frozen multi-topology signed-load exclusion experiment. Own implementation."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import csv
import fcntl
import hashlib
import json
import re
import time
from tempfile import gettempdir

import numpy as np
from pathlib import Path
REPOSITORY = Path(__file__).resolve().parent
case_directory = REPOSITORY / 'cases'
AL = np.linspace(.2, 1., 81)
PR = json.loads((REPOSITORY / 'protocol.json').read_text())
def load(name):
 txt = (case_directory / f'{name}.m').read_text()
 def mat(k):
  raw=re.search(r'mpc\.'+k+r'\s*=\s*\[(.*?)\];',txt,re.S).group(1)
  return np.array([[float(v) for v in l.split('%')[0].strip().rstrip(';').split()] for l in raw.splitlines() if l.split('%')[0].strip()])
 bus=mat('bus');br=mat('branch');base=float(re.search(r'mpc.baseMVA\s*=\s*([\d.]+)',txt).group(1));zb=bus[0,9]**2/base
 ed=np.c_[br[:,:2]-1,br[:,2:4]/zb];normal=list(np.flatnonzero(br[:,10]));ties=list(np.flatnonzero(br[:,10]==0))
 if not ties:
  extra=np.array([[10,42,.8/zb,.8/zb],[14,45,.8/zb,.8/zb],[26,64,.8/zb,.8/zb],[49,58,.8/zb,.8/zb],[19,68,.8/zb,.8/zb]])
  ties=list(range(len(ed),len(ed)+len(extra)));ed=np.vstack([ed,extra])
 return bus,ed,normal,ties,base

def setup(name,ids):
 bus,ed,_,_,base=load(name);n=len(bus)-1;adj=[[] for _ in bus]
 for ei in ids:
  a,b=ed[ei,:2].astype(int);adj[a].append((b,ei));adj[b].append((a,ei))
 order=[0];parold={0:-1};edgeold={}
 for a in order:
  for b,ei in adj[a]:
   if b not in parold:parold[b]=a;edgeold[b]=ei;order.append(b)
 if len(order)!=n+1:return None
 inv={v:k for k,v in enumerate(order)};par=np.array([inv[parold[v]] for v in order[1:]])
 A=np.zeros((n,n))
 for i in range(n):
  j=i+1
  while j:A[i,j-1]=1;j=par[j-1]
 D=A.T;r=np.array([ed[edgeold[v],2] for v in order[1:]]);x=np.array([ed[edgeold[v],3] for v in order[1:]])
 H=2*A@(np.diag(r)@D@np.diag(r)+np.diag(x)@D@np.diag(x))-A@np.diag(r*r+x*x)
 assert H.min()>-1e-12
 return dict(name=name,n=n,A=A,D=D,r=r,x=x,H=H,z=r+1j*x,par=par,p=bus[order[1:],2]/(base*1000),q=bus[order[1:],3]/(base*1000),ids=ids,order=order)

def menus(name):
 bus,ed,normal,ties,base=load(name);out=[normal]
 for tie in ties:
  for cut in normal:
   ids=sorted((set(normal)|{tie})-{cut})
   if setup(name,ids) is not None:out.append(ids)
 return out

def scenarios(N,rng,ti):
 out=[];n=N['n']
 for j in range(PR['cases_per_topology']):
  group=j%3;load=N['p']*rng.uniform(.7,1.5,n);q=N['q']*rng.uniform(.7,1.3,n);pv=np.zeros(n)
  nodes=rng.choice(np.arange(n),size=6,replace=False);pv[nodes]=rng.dirichlet(np.ones(6))*load.sum()*rng.uniform(*[(0.,.4),(.7,1.5),(1.5,2.7)][group])
  imax=np.maximum(.035,np.abs(N['D']@(N['p']+1j*N['q']))*rng.uniform(1.1,1.8,n))
  out.append(dict(id=f'{N["name"]}_{ti}_{j}',topology=f'{N["name"]}_{ti}',group=group,load=load,q=q,pv=pv,imax=imax,vs=float(rng.uniform(.995,1.025))))
 return out

def ac(N,c,aa):
 a=np.atleast_1d(aa);s=a[:,None]*(c['load']+1j*c['q'])[None,:]-c['pv'][None,:];V=np.full_like(s,c['vs'],dtype=complex)
 for _ in range(300):
  J=np.conj(s/V)@N['D'].T;nv=c['vs']-(J*N['z'])@N['A'].T;err=np.max(np.abs(nv-V),axis=1);V=nv
  if err.max()<1e-11:break
 J=np.conj(s/V)@N['D'].T;vm=np.abs(V);im=np.abs(J)
 g=(err<1e-8)&(vm.min(1)>=.95-1e-8)&(vm.max(1)<=1.05+1e-8)&np.all(im<=c['imax']*(1+1e-8),axis=1)
 return g,vm,im,err

def ms(l,u):return np.where(l>0,l*l,np.where(u<0,u*u,0.))
def keep(N,c,lo,hi,K):
 lo=np.atleast_1d(lo)[:,None];hi=np.atleast_1d(hi)[:,None];D,A,r,x,H=N['D'],N['A'],N['r'],N['x'],N['H']
 fpL=(lo*c['load']-c['pv'])@D.T;fpU=(hi*c['load']-c['pv'])@D.T;fqL=(lo*c['q'])@D.T;fqU=(hi*c['q'])@D.T
 cL=2*(fpL*r+fqL*x)@A.T;cU=2*(fpU*r+fqU*x)@A.T;ll=np.zeros_like(fpL);lu=np.broadcast_to(c['imax']**2,ll.shape).copy();valid=np.ones(len(lo),bool)
 for _ in range(K):
  PL=fpL+(ll*r)@D.T;PU=fpU+(lu*r)@D.T;QL=fqL+(ll*x)@D.T;QU=fqU+(lu*x)@D.T
  vl=c['vs']**2-cU-lu@H.T;vu=c['vs']**2-cL-ll@H.T
  valid &=np.all(vu>=.95**2-1e-9,1)&np.all(vl<=1.05**2+1e-9,1)
  vl=np.maximum(vl,.95**2);vu=np.minimum(vu,1.05**2)
  vpl=np.column_stack([np.full(len(lo),c['vs']**2),vl])[:,N['par']];vpu=np.column_stack([np.full(len(lo),c['vs']**2),vu])[:,N['par']]
  nl=np.maximum(ll,(ms(PL,PU)+ms(QL,QU))/vpu);nu=np.minimum(lu,(np.maximum(PL*PL,PU*PU)+np.maximum(QL*QL,QU*QU))/vpl)
  valid &= np.all(nl<=nu+1e-9,1);ll,lu=nl,nu
 return valid

def mask(N,c,m):
 if m.startswith('point'):return keep(N,c,AL,AL,int(m[-1]))
 if m=='block8':
  starts=np.arange(0,len(AL),8);ends=np.minimum(starts+7,len(AL)-1);good=keep(N,c,AL[starts],AL[ends],1)
  return np.repeat(good,8)[:len(AL)]
 return np.ones(len(AL),bool)

def search(N,c,m):
 if m=='descending_batch8':
  calls=0
  for end in range(len(AL),0,-8):
   a=AL[max(0,end-8):end];calls+=len(a);g=ac(N,c,a)[0]
   if g.any():return float(a[np.flatnonzero(g)[-1]]),calls
  return None,calls
 a=AL[mask(N,c,m)]
 if not len(a):return None,0
 g=ac(N,c,a)[0];return float(a[np.flatnonzero(g)[-1]]) if g.any() else None,len(a)

def dump(output_directory,name,rows):
 with (output_directory/name).open('w', newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def run(split, output_directory):
 output_directory.mkdir(parents=True, exist_ok=True)
 if split == 'confirmation':
  frozen = json.loads((REPOSITORY / 'frozen_selection.json').read_text())
  assert hashlib.sha256((REPOSITORY / 'protocol.json').read_bytes()).hexdigest() == frozen['protocol_sha256']
 rng=np.random.default_rng(PR['seed_'+split]);allnet=[];inputs=[];setuprows=[]
 for name in PR['networks']:
  menu=menus(name);order=np.random.default_rng(20261114304).permutation(len(menu));idx=order[:3] if split=='development' else order[3:12]
  for ti in idx:
   tt=time.perf_counter();N=setup(name,menu[ti]);secs=time.perf_counter()-tt;cases=scenarios(N,rng,int(ti));allnet.append((N,cases));setuprows.append(dict(topology=cases[0]['topology'],seconds=secs,network=name,ids=menu[ti]));inputs.extend(cases)
 (output_directory/f'{split}_inputs.json').write_text(json.dumps(inputs,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x));(output_directory/f'{split}_topologies.json').write_text(json.dumps(setuprows,default=lambda x:int(x)))
 oracle=[];refs={};nonmono=0
 for N,cs in allnet:
  for c in cs:
   g,v,i,err=ac(N,c,AL);refs[c['id']]=float(AL[np.flatnonzero(g)[-1]]) if g.any() else None
   nonmono+=int(any(g[j] and not all(g[:j]) for j in range(1,len(g))))
   for m in ['point1','point3','block8']:assert not np.any(g&~mask(N,c,m)),(c['id'],m)
   for j,a in enumerate(AL):oracle.append(dict(id=c['id'],alpha=a,feasible=int(g[j]),vmin=v[j].min(),vmax=v[j].max(),current_ratio=(i[j]/c['imax']).max(),residual=err[j]))
 dump(output_directory,f'{split}_oracle.csv',oracle);rows=[]
 with (Path(gettempdir()) / 'distribution-restoration-screening.compute.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  for rep in range(PR['repetitions']):
   for N,cs in allnet:
    for c in cs:
     for m in rng.permutation(PR['methods']):
      t=time.perf_counter();ans,calls=search(N,c,m);sec=time.perf_counter()-t;assert ans==refs[c['id']]
      rows.append(dict(id=c['id'],topology=c['topology'],network=N['name'],group=c['group'],rep=rep,method=m,seconds=sec,AC_states=calls,answer=ans))
 dump(output_directory,f'{split}_timings.csv',rows)
 metrics=dict(topologies=len(allnet),profiles=len(inputs),AC_oracle_states=len(oracle),non_downward_closed_profiles=nonmono,decision_mismatches=0,false_exclusions=0,static_setup_s=sum(r['seconds'] for r in setuprows),methods={})
 for m in PR['methods']:
  times={c['id']:float(np.median([r['seconds'] for r in rows if r['id']==c['id'] and r['method']==m])) for c in inputs}
  metrics['methods'][m]=dict(total_s=sum(times.values()),AC_states=sum(r['AC_states'] for r in rows if r['method']==m and r['rep']==0),topology_s={x['topology']:sum(times[c['id']] for c in inputs if c['topology']==x['topology']) for x in setuprows},by_group_s={g:sum(times[c['id']] for c in inputs if c['group']==g) for g in range(3)})
 (output_directory/f'{split}_metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2),flush=True)
 if split=='development':
  selected=min(['point1','point3','block8'],key=lambda m:metrics['methods'][m]['total_s']);(output_directory/'development_selection.json').write_text(json.dumps(dict(selected=selected,protocol_sha256=hashlib.sha256((REPOSITORY/'protocol.json').read_bytes()).hexdigest(),basis='lowest complete development time'),indent=2))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("development", "confirmation"))
    parser.add_argument("--case-dir", type=Path, default=case_directory)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "generated")
    args = parser.parse_args()
    case_directory = args.case_dir
    run(args.split, args.output_dir)
