import json, glob, re, os, sys, collections, random, math, posixpath
random.seed(0)
REG="/raid/data/shu/rft_datasets/prefix_pool/registry.json"
snaps=json.load(open(REG))["snapshots"]
L=os.path.expanduser("~/workspace/behavior-1k-solution/logs/")
files=sorted(glob.glob(L+".fleet_records_rft4[1-9]_*.jsonl")+glob.glob(L+".fleet_records_rft50_*.jsonl"))
recs=[]; unk=collections.Counter()
for f in files:
    wave=re.search(r"rft(\d+)_",f).group(1)
    for line in open(f):
        try: d=json.loads(line)
        except Exception: continue
        pr=d.get("prefix_replay")
        kind="fresh"; src=None
        if pr is not None:
            key="%s/%s"%(d.get("task_name"), posixpath.basename(pr.get("snapshot_path") or ""))
            e=snaps.get(key); src=pr.get("source_episode_id")
            kind=(e.get("capture_kind") or "none") if e else "unregistered"
            if not e: unk[wave]+=1
        recs.append(dict(wave=int(wave),task=d["task_name"],inst=d.get("instance_index"),ep=d["episode_id"],succ=1 if d["success"] else 0,q=d.get("q_score_final"),steps=d.get("num_steps"),kind=kind,src=src))
json.dump(recs,open("/tmp/planner_check/recs.json","w"))
print("records",len(recs),"files",len(files),"unregistered restarts by wave",dict(unk))
by=collections.defaultdict(lambda:[0,0])
for r in recs: by[r["kind"]][0]+=r["succ"]; by[r["kind"]][1]+=1
print("\nRAW success by kind (rft41-50):")
for k,(s,n) in sorted(by.items(), key=lambda x:-x[1][1]): print("  %-18s %5d/%5d = %.3f"%(k,s,n,s/n))

def cells(kind):
    C=collections.defaultdict(lambda:{"f":[], "r":[]})
    for r in recs:
        if r["kind"]=="fresh": C[(r["task"],r["wave"])]["f"].append(r["succ"])
        elif (kind=="all" and r["kind"]!="fresh") or r["kind"]==kind: C[(r["task"],r["wave"])]["r"].append(r["succ"])
    return {k:v for k,v in C.items() if v["f"] and v["r"]}

def matched(kind, B=4000):
    C=cells(kind)
    tasks=sorted({t for t,_ in C})
    def est(cellset, w):
        ds=[]; ws=[]
        for k in cellset:
            v=C[k]; d=sum(v["r"])/len(v["r"])-sum(v["f"])/len(v["f"])
            ds.append(d); ws.append(1.0 if w=="eq" else 1/(1/len(v["r"])+1/len(v["f"])))
        return sum(d*x for d,x in zip(ds,ws))/sum(ws)
    out={}
    bytask=collections.defaultdict(list)
    for k in C: bytask[k[0]].append(k)
    for w in ("eq","harm"):
        pt=est(list(C),w); bs=[]
        for _ in range(B):
            smp=[random.choice(tasks) for _ in tasks]
            cs=[k for t in smp for k in bytask[t]]
            bs.append(est(cs,w))
        bs.sort(); se=(sum((b-pt)**2 for b in bs)/(B-1))**.5
        out[w]=(pt,se,bs[int(.025*B)],bs[int(.975*B)])
    num=var=rd_num=rd_den=0.0
    for v in C.values():
        a=sum(v["r"]); n1=len(v["r"]); b=sum(v["f"]); n0=len(v["f"]); N=n1+n0; m1=a+b
        if N<2: continue
        num+=a-n1*m1/N; var+=n1*n0*m1*(N-m1)/(N*N*(N-1))
        w=n1*n0/N; rd_num+=w*(a/n1-b/n0); rd_den+=w
    z=num/var**.5 if var>0 else float("nan"); p=math.erfc(abs(z)/2**.5)
    nr=sum(len(v["r"]) for v in C.values()); nf=sum(len(v["f"]) for v in C.values())
    sr=sum(sum(v["r"]) for v in C.values()); sf=sum(sum(v["f"]) for v in C.values())
    print("\n%s: %d task-wave cells, %d tasks; restarts %d/%d, fresh %d/%d in those cells"%(kind,len(C),len(tasks),sr,nr,sf,nf))
    for w,(pt,se,lo,hi) in out.items(): print("  cell-mean dSR (%-4s weights) %+.3f ± %.3f  task-boot 95%% CI [%+.3f, %+.3f]"%(w,pt,se,lo,hi))
    print("  Mantel-Haenszel RD %+.3f; CMH z %+.2f, two-sided p %.4f"%(rd_num/rd_den,z,p))
for kind in ("qdrop_backoff","progress_backoff","progress","terminal","all"): matched(kind)
