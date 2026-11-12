"""Constraint-visible planning with explicit STOP and exact training supervision."""
from functools import lru_cache
import numpy as np
import torch
from torch import nn

from .evidence_policy import EvidenceEnvironment
from .research_policy import CubicSplineLayer, generalized_advantages


class PlanningEnvironment:
    def __init__(self, graph, task):
        self.graph, self.task = graph, task
        self.h = graph.action_dim; self.stop = self.h
        self.budget = int(task.get("budget",8))
        if not 1 <= self.budget <= self.h:
            raise ValueError("Invalid budget")
        self.base = EvidenceEnvironment(graph,max_steps=self.budget); self.base.reset(task)
        self.useful = ((self.base.initial @ graph.direct)>0) | ((graph.affiliation @ self.base.relevant)>0)
        self.allowed = self.base.allowed & self.useful
        self.pairs = np.zeros((self.h,self.h),dtype=bool)
        for a,b in task.get("forbidden_pairs",[]):
            if a in graph.hnames and b in graph.hnames:
                i,j=graph.hnames[a],graph.hnames[b];self.pairs[i,j]=self.pairs[j,i]=True
        self.observation_dim=graph.state_dim+self.h*self.h+2
        self.feature_dim=10+(self.h+1)+self.h
        self.score=lru_cache(maxsize=1024)(self._score)
        self.reset()

    def reset(self, selected=0):
        self.selected=int(selected);self.done=self.selected.bit_count()>=self.budget
        self.actions=[i for i in range(self.h) if selected & (1<<i)]
        if not self.base.initial.any():self.done=True
        return self

    def vector(self, mask):
        return np.array([(mask>>i)&1 for i in range(self.h)],dtype=np.float32)

    def _score(self, mask):
        _,_,m=self.base.coverage(self.vector(mask))
        return (m["objective"],m["symptom_evidence_coverage"],m["target_evidence_coverage"],m["ppi_edge_coverage"])

    def mask(self):
        selected=self.vector(self.selected).astype(bool)
        legal=self.allowed & ~selected & ~self.pairs[selected].any(axis=0)
        if self.selected.bit_count()>=self.budget:legal[:]=False
        return np.r_[legal,True]

    def state(self):
        chosen=self.vector(self.selected)
        self.base.selected=chosen
        observation=np.r_[self.base.state().numpy(),self.pairs.ravel().astype(np.float32),
                          (self.budget-chosen.sum())/8,self.budget/8].astype(np.float32)
        mask=self.mask();now=np.array(self.score(self.selected))
        gain=np.zeros(self.h+1,dtype=np.float32)
        changes=np.zeros((self.h,4),dtype=np.float32)
        for i in np.flatnonzero(mask[:-1]):
            changes[i]=np.array(self.score(self.selected|(1<<int(i))))-now;gain[i]=changes[i,0]
        positive=np.maximum(gain[:-1],0)
        features=[]
        for i in range(self.h+1):
            pair=self.pairs[i].astype(np.float32) if i<self.h else np.zeros(self.h,dtype=np.float32)
            change=changes[i] if i<self.h else np.zeros(4)
            scalars=[*change,(self.budget-chosen.sum())/8,chosen.sum()/8,pair.sum()/self.h,
                     float(pair@positive)/max(float(positive.sum()),1e-8),float(i==self.stop),float(mask[i])]
            features.append(np.r_[scalars,np.eye(self.h+1,dtype=np.float32)[i],pair])
        return torch.from_numpy(observation),torch.tensor(np.array(features),dtype=torch.float32),torch.from_numpy(mask)

    def step(self, action):
        if self.done or not 0<=action<=self.h or not self.mask()[action]:raise ValueError("Invalid planning action")
        if action==self.stop:
            self.done=True;return 0.
        before=self.score(self.selected)[0];self.selected|=1<<action;self.actions.append(action)
        self.done=self.selected.bit_count()>=self.budget or not self.mask()[:-1].any()
        return self.score(self.selected)[0]-before

    def exact_table(self):
        ids=np.flatnonzero(self.allowed);n=len(ids)
        all_masks,all_scores=[],[];g=self.graph;e=self.base
        tw=.3 if e.relevant.any() else 0;pw=.05 if len(e.relevant_edges) else 0
        for start in range(0,1<<n,4096):
            integers=np.arange(start,min(start+4096,1<<n),dtype=np.int64)
            bits=((integers[:,None]>>np.arange(n))&1).astype(np.float32)
            bits=bits[bits.sum(axis=1)<=self.budget]
            if not len(bits):continue
            selected=np.zeros((len(bits),self.h),dtype=np.float32);selected[:,ids]=bits
            for a,b in zip(*np.where(np.triu(self.pairs,1))):
                selected=selected[~((selected[:,a]>0)&(selected[:,b]>0))]
            if not len(selected):continue
            targets=(selected@g.affiliation>0).astype(np.float32)
            direct=(selected@g.direct.T>0).astype(np.float32)
            indirect=(targets@g.symptom_target.T)/np.maximum(g.symptom_target.sum(axis=1),1)
            symptom=np.maximum(direct,indirect)@e.initial/max(float(e.initial.sum()),1e-8)
            target=targets@e.relevant/max(float(e.relevant.sum()),1)
            edges=e.relevant_edges
            ppi=(targets[:,edges[:,0]]*targets[:,edges[:,1]]).mean(axis=1) if len(edges) else 0
            scores=(.65*symptom+tw*target+pw*ppi)/(.65+tw+pw)-.02*selected.sum(axis=1)
            masks=(selected.astype(np.int64)*(1<<np.arange(self.h))).sum(axis=1)
            all_masks.append(masks);all_scores.append(scores)
        return np.concatenate(all_masks),np.concatenate(all_scores)


def greedy_action(env):
    _,features,mask=env.state();gain=features[:,0].numpy().copy();gain[~mask.numpy()]=-np.inf
    # Prefer STOP over a zero-gain herb; otherwise stable herb index ties.
    return env.stop if np.max(gain[:-1])<=1e-7 else int(np.argmax(gain))


def lookahead_action(env):
    mask=env.mask();before=env.score(env.selected)[0];best=0.;action=env.stop
    for a in np.flatnonzero(mask[:-1]):
        selected=env.selected|(1<<int(a));value=env.score(selected)[0]-before
        if selected.bit_count()<env.budget:
            chosen=env.vector(selected).astype(bool)
            choices=env.allowed & ~chosen & ~env.pairs[chosen].any(axis=0)
            extra=max([0.,*[env.score(selected|(1<<int(b)))[0]-env.score(selected)[0] for b in np.flatnonzero(choices)]])
            value+=extra
        if value>best+1e-7:best,action=value,int(a)
    return action


class ActionScorer(nn.Module):
    def __init__(self,observation_dim,feature_dim,kind="kan",prior=True):
        super().__init__();self.prior=prior
        self.context=nn.Sequential(nn.Linear(observation_dim,16),nn.Tanh())
        if kind=="kan":
            self.network=nn.Sequential(CubicSplineLayer(feature_dim+16,16),nn.LayerNorm(16),nn.Tanh(),CubicSplineLayer(16,1))
            nn.init.zeros_(self.network[-1].base);nn.init.zeros_(self.network[-1].coefficients);nn.init.zeros_(self.network[-1].bias)
        else:
            self.network=nn.Sequential(nn.Linear(feature_dim+16,192),nn.Tanh(),nn.Linear(192,1))
            nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,observation,features):
        context=self.context(observation)[:,None,:].expand(-1,features.shape[1],-1)
        inputs=torch.cat([features,context],dim=-1)
        residual=self.network(inputs.reshape(-1,inputs.shape[-1])).reshape(features.shape[:2])
        return (features[:,:,0]+features[:,:,8]*1e-7 if self.prior else 0)+.1*residual


class PlanningPolicy(nn.Module):
    def __init__(self,observation_dim,feature_dim,variant="residual_kan",temperature=.03):
        super().__init__();self.temperature=temperature
        self.actor=ActionScorer(observation_dim,feature_dim,"mlp" if variant=="residual_mlp" else "kan",variant!="kan_without_prior")
        self.critic=nn.Sequential(nn.Linear(observation_dim,64),nn.Tanh(),nn.Linear(64,1))

    def distribution(self,observation,features,mask):
        scores=self.actor(observation,features)/self.temperature
        return torch.distributions.Categorical(logits=scores.masked_fill(~mask,-1e9)),self.critic(observation).squeeze(-1)


def rollout(env,policy=None,method="greedy"):
    env.reset()
    while not env.done:
        if policy is not None:
            observation,features,mask=env.state()
            with torch.no_grad():distribution,_=policy.distribution(observation[None],features[None],mask[None])
            action=int(distribution.probs.argmax())
        else:action=lookahead_action(env) if method=="lookahead2" else greedy_action(env)
        env.step(action)
    return env.selected,env.score(env.selected)[0]


def training_states(env,seed=42):
    masks,scores=env.exact_table();bestmask=int(masks[int(np.argmax(scores))])
    states=set();env.reset()
    while not env.done:
        states.add(env.selected);env.step(greedy_action(env))
    env.reset()
    while not env.done:
        states.add(env.selected)
        candidates=[i for i in range(env.h) if bestmask&(1<<i) and not env.selected&(1<<i)]
        if not candidates:env.step(env.stop);break
        env.step(max(candidates,key=lambda i:(env.score(env.selected|(1<<i))[0],-i)))
    possible=masks[np.array([int(m).bit_count()<env.budget for m in masks])]
    rng=np.random.default_rng(seed)
    if len(possible):states.update(int(x) for x in rng.choice(possible,min(2,len(possible)),replace=False))
    output=[]
    for selected in sorted(states):
        env.reset(selected);observation,features,mask=env.state();current=env.score(selected)[0]
        q=np.zeros(env.h+1,dtype=np.float32)
        for action in np.flatnonzero(mask.numpy()[:-1]):
            required=selected|(1<<int(action));compatible=(masks & required)==required
            assert compatible.any();q[action]=float(np.max(scores[compatible]))-current
        output.append((observation,features,mask,torch.from_numpy(q),float(q[mask.numpy()].max())))
    return output,{"feasible_subsets":len(masks),"states":len(states)}


class PlannerPPO:
    def __init__(self,policy,actor_lr=1e-4,critic_lr=3e-4,target_kl=.02):
        self.policy=policy;self.actor=torch.optim.Adam(policy.actor.parameters(),lr=actor_lr)
        self.critic=torch.optim.Adam(policy.critic.parameters(),lr=critic_lr);self.target_kl=target_kl

    def collect(self,envs,rng,episodes=24):
        rows=[];outcomes=[]
        for _ in range(episodes):
            env=envs[int(rng.integers(len(envs)))];env.reset();episode=[]
            while not env.done:
                obs,feat,mask=env.state()
                with torch.no_grad():distribution,value=self.policy.distribution(obs[None],feat[None],mask[None]);action=distribution.sample()
                reward=env.step(int(action));episode.append((obs,feat,mask,action[0],distribution.log_prob(action)[0],distribution.probs[0],float(value),reward))
            outcomes.append(env.score(env.selected)[0])
            adv=generalized_advantages([r[-1] for r in episode],[r[-2] for r in episode],gamma=1.,lam=.95)
            rows.extend((*r[:6],a,a+r[-2]) for r,a in zip(episode,adv))
        if not rows:raise ValueError("No training transitions")
        batch=[torch.stack([r[i] for r in rows]) for i in range(6)]
        advantages=torch.tensor([r[6] for r in rows]);advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
        return (*batch,advantages,torch.tensor([r[7] for r in rows]),float(np.mean(outcomes)))

    def update(self,batch,entropy=.003):
        obs,feat,mask,actions,oldlog,oldprob,adv,returns,mean=batch;updates=0
        for _ in range(4):
            dist,_=self.policy.distribution(obs,feat,mask)
            kl=(oldprob*(oldprob.clamp_min(1e-12).log()-dist.probs.clamp_min(1e-12).log())).sum(-1).mean()
            if float(kl.detach())>self.target_kl:break
            ratio=(dist.log_prob(actions)-oldlog).exp()
            loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()-entropy*dist.entropy().mean()
            self.actor.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.policy.actor.parameters(),.5);self.actor.step();updates+=1
        for _ in range(4):
            value=self.policy.critic(obs).squeeze(-1);loss=(value-returns).square().mean()
            self.critic.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.policy.critic.parameters(),.5);self.critic.step()
        return {"actor_updates":updates,"last_checked_kl":float(kl.detach()),"critic_mse":float(loss.detach()),"sampled_objective":mean}
