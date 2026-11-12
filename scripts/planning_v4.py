"""Frozen planning study: exact train supervision, residual networks and PPO."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from dmease.evidence_policy import EvidenceGraph,canonical_hash
from dmease.experiment_metrics import paired_statistics,holm_adjust
from dmease.research_policy import refine_selection
from dmease.planning_v4 import PlanningEnvironment,PlanningPolicy,PlannerPPO,rollout,training_states

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"experiments/v4";PRIVATE=ROOT/"outputs/research_v4/planning"


def read(path):return json.loads(path.read_text(encoding="utf8"))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8",newline="\n")


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(protocol,graph):
    files=["scripts/planning_v4.py","src/dmease/planning_v4.py","src/dmease/evidence_policy.py","src/dmease/research_policy.py","src/dmease/experiment_metrics.py"]
    return {"protocol":canonical_hash(protocol),"evidence":graph.fingerprint,"torch":str(torch.__version__),"numpy":np.__version__,
            "device":"cpu","threads":1,"source_lf_sha256":{s:hashlib.sha256((ROOT/s).read_text(encoding="utf8").encode()).hexdigest() for s in files}}


def prepare(protocol,graph,stamp):
    used=set()
    for filename in ("experiments/queries.json","experiments/v2/queries.json"):
        for rows in read(ROOT/filename)["splits"].values():
            used.update(canonical_hash(sorted(r["symptoms"])) for r in rows)
    bases=[]
    for count in (3,4,5):
        for names in itertools.combinations(sorted(graph.snames),count):
            identity=canonical_hash(sorted(names))
            if identity not in used:bases.append((identity,names))
    bases.sort(key=lambda x:hashlib.sha256(("v4-planner:"+x[0]).encode()).hexdigest())
    splits={};start=0
    for label,count in (("train",320),("validation",64),("test",128)):
        tasks=[]
        for base_id,names in bases[start:start+count]:
            rng=np.random.default_rng(int(base_id[:8],16))
            herbs=[h["name"] for h in graph.herbs]
            excluded=rng.choice(herbs,2,replace=False).tolist()
            remaining=[h for h in herbs if h not in excluded]
            paired=rng.choice(remaining,4,replace=False).tolist()
            for profile in ("standard","requested_constraints"):
                task={"base_id":base_id,"symptoms":dict.fromkeys(names,1.),"profile":profile,
                      "forbidden":excluded if profile=="requested_constraints" else [],
                      "forbidden_pairs":[paired[:2],paired[2:]] if profile=="requested_constraints" else [],
                      "budget":4 if profile=="requested_constraints" else 8,
                      "scope":"engineering evidence-retrieval request, not a patient or a pharmacological contraindication"}
                task["id"]=canonical_hash(task);tasks.append(task)
        splits[label]=tasks;start+=count
    save(OUT/"planning_queries.json",{"provenance":stamp,"splits":splits,"excluded_previous_base_sets":len(used)})
    tensors=[[],[],[],[],[]];audit=[]
    for i,task in enumerate(splits["train"]):
        env=PlanningEnvironment(graph,task);rows,metadata=training_states(env,seed=int(task["id"][:8],16))
        for row in rows:
            for j,value in enumerate(row):tensors[j].append(value)
        audit.append({"query_id":task["id"],**metadata})
        if (i+1)%40==0:print(f"Exact train teacher {i+1}/{len(splits['train'])}",flush=True)
    data=[torch.stack(tensors[i]) for i in range(4)]+[torch.tensor(tensors[4])]
    PRIVATE.mkdir(parents=True,exist_ok=True);torch.save({"tensors":data,"provenance":stamp},PRIVATE/"teacher.pt")
    save(OUT/"planning_teacher.json",{"provenance":stamp,"rows":audit,"states":len(data[0]),"checkpoint_sha256":sha(PRIVATE/"teacher.pt")})
    print("Teacher ready",len(data[0]),flush=True)


def evaluate(envs,policy=None,method="greedy"):
    return float(np.mean([rollout(env,policy,method)[1] for env in envs]))


def model_for(env,variant,config):
    return PlanningPolicy(env.observation_dim,env.feature_dim,variant,config["temperature"])


def train_one(graph,tasks,protocol,stamp,variant,seed,data):
    torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed);config=protocol["training"]
    envs=[PlanningEnvironment(graph,t) for t in tasks["train"]]
    validation=[PlanningEnvironment(graph,t) for t in tasks["validation"]]
    policy=model_for(envs[0],variant,config)
    history=[];best=-np.inf;chosen=-1;best_state=None
    def observe(iteration):
        nonlocal best,chosen,best_state
        policy.eval();score=evaluate(validation,policy)
        history.append({"iteration":iteration,"objective":score})
        if score>best+1e-8:
            best,chosen=score,iteration;best_state={k:v.detach().clone() for k,v in policy.state_dict().items()}
        return score
    observe(-1)
    obs,feat,mask,q,values=data
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=3e-4)
    for epoch in range(config["warmup_epochs"]):
        policy.train()
        for index in torch.randperm(len(obs)).split(64):
            prediction=policy.actor(obs[index],feat[index]).masked_fill(~mask[index],-1e9)
            target=q[index].masked_fill(~mask[index],-1e9)
            teacher=torch.softmax(target/config["temperature"],dim=-1)
            loss=-(teacher*torch.log_softmax(prediction/config["temperature"],dim=-1)).sum(-1).mean()
            pc=prediction-prediction.max(-1,keepdim=True).values
            tc=target-target.max(-1,keepdim=True).values
            loss=loss+.5*((pc-tc).masked_fill(~mask[index],0)/.1).square().sum(-1).div(mask[index].sum(-1)).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);optimizer.step()
    critic=torch.optim.Adam(policy.critic.parameters(),lr=config["critic_lr"])
    for _ in range(config["critic_epochs"]):
        for index in torch.randperm(len(obs)).split(128):
            loss=(policy.critic(obs[index]).squeeze(-1)-values[index]).square().mean()
            critic.zero_grad();loss.backward();critic.step()
    warm_score=observe(0)
    checkpoint=PRIVATE/f"{variant}_{seed}.pt";warm=PRIVATE/f"{variant}_{seed}_warm.pt"
    torch.save({"state_dict":policy.state_dict(),"provenance":stamp},warm)
    trainer=PlannerPPO(policy,config["actor_lr"],config["critic_lr"],config["target_kl"])
    diagnostics=[]
    for iteration in range(1,config["ppo_iterations"]+1):
        policy.train();diag=trainer.update(trainer.collect(envs,rng,config["episodes"]),config["initial_entropy"]*(1-iteration/config["ppo_iterations"]))
        diagnostics.append({"iteration":iteration,**diag})
        if iteration%config["validation_interval"]==0:
            value=observe(iteration);print(f"{variant}/{seed} iteration {iteration}: validation {value:.6f}",flush=True)
    torch.save({"state_dict":best_state,"provenance":stamp},checkpoint)
    return {"provenance":stamp,"variant":variant,"seed":seed,"selected_iteration":chosen,"validation_objective":best,
            "warm_validation_objective":warm_score,"history":history,"ppo_diagnostics":diagnostics,
            "actor_parameters":sum(p.numel() for p in policy.actor.parameters()),"critic_parameters":sum(p.numel() for p in policy.critic.parameters()),
            "checkpoint_sha256":sha(checkpoint),"warm_checkpoint_sha256":sha(warm),"frozen_before_test_at":datetime.now(timezone.utc).isoformat()}


def result_row(env,mask,seconds):
    selected={env.graph.herbs[i]["name"] for i in range(env.h) if mask&(1<<i)}
    violations=len(selected&set(env.task.get("forbidden",[])))+sum(a in selected and b in selected for a,b in env.task.get("forbidden_pairs",[]))+int(len(selected)>env.budget)
    metrics=env.score(mask)
    return {"query_id":env.task["id"],"base_id":env.task["base_id"],"profile":env.task["profile"],"objective":metrics[0],
            "symptom_coverage":metrics[1],"target_coverage":metrics[2],"ppi_coverage":metrics[3],"selected_herbs":sorted(selected),
            "herb_count":len(selected),"hard_violations":violations,"seconds":seconds}


def aggregate(graph,tasks,protocol,stamp):
    selections={}
    for variant in protocol["variants"]:
        for seed in protocol["seeds"]:
            selected=read(OUT/f"planning_training/{variant}_{seed}.json");assert selected["provenance"]==stamp
            assert sha(PRIVATE/f"{variant}_{seed}.pt")==selected["checkpoint_sha256"]
            assert sha(PRIVATE/f"{variant}_{seed}_warm.pt")==selected["warm_checkpoint_sha256"]
            selections[variant,seed]=selected
    envs=[PlanningEnvironment(graph,t) for t in tasks["test"]]
    methods={name:[] for name in ("greedy","lookahead2","greedy_search","exact")}
    for name in methods:
        rows=[]
        for env in envs:
            env.score.cache_clear();start=time.perf_counter()
            if name=="exact":
                masks,values=env.exact_table();selected=int(masks[np.argmax(values)])
            else:
                selected,_=rollout(env,method="lookahead2" if name=="lookahead2" else "greedy")
                if name=="greedy_search":
                    actions,_=refine_selection(graph,env.task,[i for i in range(env.h) if selected&(1<<i)],max_steps=env.budget)
                    selected=sum(1<<i for i in actions)
            rows.append(result_row(env,selected,time.perf_counter()-start))
        methods[name].append({"seed":None,"rows":rows})
        print(name,"test",np.mean([r["objective"] for r in rows]),flush=True)
    for variant in protocol["variants"]:
        for suffix in ("","_warm","_guarded","_search"):methods[variant+suffix]=[]
        for seed in protocol["seeds"]:
            policy=model_for(envs[0],variant,protocol["training"])
            for mode in ("warm","selected"):
                path=PRIVATE/f"{variant}_{seed}{'_warm' if mode=='warm' else ''}.pt"
                payload=torch.load(path,map_location="cpu",weights_only=True);policy.load_state_dict(payload["state_dict"]);policy.eval()
                raw,guarded,searched=[],[],[]
                for env,greedy in zip(envs,methods["greedy"][0]["rows"]):
                    env.score.cache_clear();start=time.perf_counter();selected,_=rollout(env,policy)
                    elapsed=time.perf_counter()-start;raw.append(result_row(env,selected,elapsed))
                    if mode=="selected":
                        baseline=sum(1<<graph.hnames[h] for h in greedy["selected_herbs"])
                        chosen=selected if env.score(selected)[0]>greedy["objective"]+1e-7 else baseline
                        guarded.append(result_row(env,chosen,elapsed+greedy["seconds"]))
                        begin=time.perf_counter();actions,_=refine_selection(graph,env.task,[i for i in range(env.h) if selected&(1<<i)],max_steps=env.budget)
                        searched.append(result_row(env,sum(1<<i for i in actions),elapsed+time.perf_counter()-begin))
                methods[variant+('_warm' if mode=='warm' else '')].append({"seed":seed,"rows":raw})
                if mode=="selected":
                    methods[variant+"_guarded"].append({"seed":seed,"rows":guarded});methods[variant+"_search"].append({"seed":seed,"rows":searched})
                print(variant,seed,mode,"test",np.mean([r["objective"] for r in raw]),flush=True)
    optimum={r["query_id"]:r["objective"] for r in methods["exact"][0]["rows"]}
    summary={};mean_base={}
    for method,runs in methods.items():
        summary[method]={}
        for profile in ("all","standard","requested_constraints"):
            rows=[r for run in runs for r in run["rows"] if profile=="all" or r["profile"]==profile]
            summary[method][profile]={"evaluated_rows_with_seeds":len(rows),"objective":float(np.mean([r["objective"] for r in rows])),
                "herb_count":float(np.mean([r["herb_count"] for r in rows])),"hard_violations":sum(r["hard_violations"] for r in rows),
                "mean_gap_to_optimal":float(np.mean([optimum[r["query_id"]]-r["objective"] for r in rows])),
                "optimal_rate":float(np.mean([abs(optimum[r["query_id"]]-r["objective"])<1e-6 for r in rows])),
                "median_rollout_ms":1000*float(np.median([r["seconds"] for r in rows]))}
        bases=sorted({r["base_id"] for r in runs[0]["rows"]})
        mean_base[method]=[{"query_id":base,"objective":float(np.mean([r["objective"] for run in runs for r in run["rows"] if r["base_id"]==base]))} for base in bases]
    comparisons={v:paired_statistics(mean_base[v],mean_base["greedy"]) for v in protocol["variants"]}
    for name,p in zip(comparisons,holm_adjust([r["sign_flip_p"] for r in comparisons.values()])):comparisons[name]["holm_p"]=p
    report={"provenance":stamp,"methods":methods,"summary":summary,"raw_vs_greedy_base_clustered":comparisons,
            "selected_iterations":{v:[selections[v,s]["selected_iteration"] for s in protocol["seeds"]] for v in protocol["variants"]},
            "warm_to_selected":{v:paired_statistics(mean_base[v],mean_base[v+"_warm"]) for v in protocol["variants"]},
            "completed_at":datetime.now(timezone.utc).isoformat()}
    save(OUT/"planning_results.json",report)
    save(OUT/"planning_summary.json",{k:v for k,v in report.items() if k!="methods"})
    print(json.dumps({k:v["all"] for k,v in summary.items()},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");parser.add_argument("--train",action="store_true");parser.add_argument("--aggregate",action="store_true");parser.add_argument("--seed",type=int)
    args=parser.parse_args();torch.set_num_threads(1)
    protocol=read(OUT/"planning_protocol.json");graph=EvidenceGraph(ROOT/"data/processed/evidence_bundle.json")
    assert graph.fingerprint==protocol["evidence_fingerprint"];stamp=provenance(protocol,graph)
    if args.prepare:prepare(protocol,graph,stamp);return
    queries=read(OUT/"planning_queries.json");assert queries["provenance"]==stamp
    if args.aggregate:aggregate(graph,queries["splits"],protocol,stamp);return
    if not args.train:parser.error("Choose --prepare, --train or --aggregate")
    teacher=read(OUT/"planning_teacher.json");assert teacher["provenance"]==stamp and sha(PRIVATE/"teacher.pt")==teacher["checkpoint_sha256"]
    data=torch.load(PRIVATE/"teacher.pt",map_location="cpu",weights_only=True)["tensors"]
    seeds=[args.seed] if args.seed is not None else protocol["seeds"]
    assert set(seeds)<=set(protocol["seeds"])
    for seed in seeds:
        for variant in protocol["variants"]:
            path=OUT/f"planning_training/{variant}_{seed}.json"
            if path.exists():
                previous=read(path)
                assert previous["provenance"]==stamp
                if (PRIVATE/f"{variant}_{seed}.pt").exists() and (PRIVATE/f"{variant}_{seed}_warm.pt").exists():
                    assert sha(PRIVATE/f"{variant}_{seed}.pt")==previous["checkpoint_sha256"]
                    continue
            started=time.perf_counter();result=train_one(graph,queries["splits"],protocol,stamp,variant,seed,data)
            result["training_seconds"]=time.perf_counter()-started;save(path,result)


if __name__=="__main__":main()
