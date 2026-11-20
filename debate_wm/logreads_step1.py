import json, sys, itertools, random
from collections import defaultdict, Counter
L="ABCDEFGHIJ"
def hs(seat): return [i for i in range(4) if i!=seat]
def maj(a):
    c=Counter(x for x in a if x); 
    if not c: return None
    x,k=c.most_common(1)[0]; return x if k>=2 else None
def rounds(d):
    h=hs(d["seat"]); R=[[d["history"][r][i]["answer"] for i in h] for r in range(3)]
    R.append([x["answer"] for x in d["final"]]); return R
def boot(vals,n=2000,seed=0):
    rng=random.Random(seed); m=sum(vals)/len(vals)
    b=sorted(sum(vals[rng.randrange(len(vals))] for _ in vals)/len(vals) for _ in range(n))
    return m,b[50],b[1949]
run=sys.argv[1]
st=json.load(open(run+"/state.json"))
D=[d for d in st["debates"] if d.get("final")]
by=defaultdict(dict)  # cond -> (t,s) -> d
for d in D: by[d["cond"]][(d["task_id"],d["seed"])]=d
conds=list(by); seeds=sorted({d["seed"] for d in D})
out={"run":run,"conds":conds,"seeds":seeds}
# A: round-1 / round-2 identity of honest turns vs none
A={}
if "none" in by:
    for c in conds:
        if c=="none": continue
        res={}
        for r in (0,1):
            same=tot=0
            for k,d in by[c].items():
                n=by["none"].get(k)
                if not n: continue
                for i in hs(d["seat"]):
                    tot+=1; same+= (d["history"][r][i]==n["history"][r][i])
            res[f"r{r+1}_identical"]=same/tot
        A[c]=res
out["A_identity_vs_none"]=A
# B: per-condition switch rates by round and 3-way split; agent vs debate accuracy
B={}
for c in conds:
    sw=defaultdict(list); split=[]; agent=[]; deb=[]; dl=[]
    for d in by[c].values():
        R=rounds(d); g=d["gold"]
        for r in range(1,4):
            for i in range(3): sw[r].append(R[r][i]!=R[r-1][i])
        split.append(len(set(R[3]))==3)
        agent.append(sum(a==g for a in R[3])/3); m=maj(R[3]); deb.append(m==g); dl.append(m is None)
    allsw=[x for v in sw.values() for x in v]
    B[c]={"switch_all":sum(allsw)/len(allsw),**{f"switch_r{r+1}":sum(v)/len(v) for r,v in sw.items()},
          "split3_final":sum(split)/len(split),"deadlock":sum(dl)/len(dl),
          "acc_agent":sum(agent)/len(agent),"acc_majority":sum(deb)/len(deb),"n":len(agent)}
out["B_rates"]=B
# C: churn. outcome ok = majority correct
def ok(d): return maj(rounds(d)[3])==d["gold"]
C={}
tasks=sorted({k[0] for k in by.get("none",{})})
if "none" in by and len(seeds)>1:
    def trans(ca,cb,same_seed):
        f=[];r=[]
        for t in tasks:
            for si in seeds:
                for sj in seeds:
                    if (si==sj)!=same_seed: continue
                    if ca==cb and si==sj: continue
                    a=by[ca].get((t,si)); b=by[cb].get((t,sj))
                    if not a or not b: continue
                    (f if ok(a) else r).append((t, (not ok(b)) if ok(a) else ok(b)))
        def agg(x):
            q=defaultdict(list)
            for t,v in x: q[t].append(v)
            return len(x), sum(v for _,v in x)/max(1,len(x))
        return {"n_ok":agg(f)[0],"f(ok->notok)":agg(f)[1],"n_notok":agg(r)[0],"r(notok->ok)":agg(r)[1]}
    C["none_vs_none_crossseed"]=trans("none","none",False)
    for c in conds:
        if c=="none": continue
        C[c+"_vs_none_sameseed"]=trans("none",c,True)
        C[c+"_vs_none_crossseed"]=trans("none",c,False)
    # agent-level churn too: per-agent final correctness flip across seeds (seat-agnostic: mean acc per debate)
out["C_churn"]=C
print(json.dumps(out,indent=1))
