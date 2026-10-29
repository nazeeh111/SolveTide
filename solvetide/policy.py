"""Interpretable joint forward/adjoint memory-allocation and refresh policy.

Every action allocates a fixed shared budget and predicts COMPLETE PAIR cost.
The model never predicts a physical field or changes numerical acceptance rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
import math
from pathlib import Path
from typing import Mapping
import numpy as np

FEATURES = ('operator_drift','residual_projection','previous_primal_reduction','previous_adjoint_reduction',
            'previous_primal_iterations','previous_adjoint_iterations','available_primal_rank','available_adjoint_rank','dimension')
ACTIONS = ('skip','balanced','forward','adjoint','primal_only','adjoint_only')
ALLOCATIONS = ((6,6,False),(6,6,True),(9,3,True),(3,9,True),(12,0,True),(0,12,True))


def vector(features: Mapping[str,float]) -> np.ndarray:
    values=np.array([float(features.get(k,0.0)) for k in FEATURES])
    if not np.isfinite(values).all():raise ValueError('finite policy features required')
    ix=[0,4,5,6,7,8]
    values[ix]=np.log1p(np.maximum(values[ix],0))
    return values


def action_plan(name: str, budget: int) -> dict:
    p,a,refresh=ALLOCATIONS[ACTIONS.index(name)]
    pq=int(round(budget*p/12))
    aq=budget-pq
    return {'primal_rank':pq,'adjoint_rank':aq,'primal_quota':pq,'adjoint_quota':aq,'refresh':refresh}


@dataclass
class ConstantPlan:
    action: str
    def plan(self, features, max_rank_budget):
        return action_plan(self.action,max_rank_budget)


@dataclass
class ConstantRank:
    rank: int
    def choose(self, features,max_rank):return min(self.rank,max_rank)


class RankPolicy:
    """Per-action ridge models of log complete-pair seconds.

    A 5% predicted saving over 'skip' is required. Development chronological
    validation decides enabled; confirmation can evaluate force=True without
    modifying shipped weights or gate. This is a fitted local cost heuristic,
    not a proof of sequence-optimal allocation.
    """
    def __init__(self,document:dict,*,force:bool=False):
        if document.get('schema_version')!=2 or document.get('features')!=list(FEATURES) or document.get('actions')!=list(ACTIONS):
            raise ValueError('unsupported joint policy schema')
        self.document=document
        self.mean=np.array(document['mean'],dtype=float)
        self.scale=np.array(document['scale'],dtype=float)
        self.weights=np.array(document['weights'],dtype=float)
        if self.mean.shape!=(9,) or self.scale.shape!=(9,) or self.weights.shape!=(6,10):raise ValueError('policy shape mismatch')
        if not all(np.isfinite(a).all() for a in (self.mean,self.scale,self.weights)) or np.any(self.scale<=0):raise ValueError('invalid policy weights')
        self.margin=float(document.get('refresh_margin',.05))
        if not math.isfinite(self.margin) or not 0<=self.margin<1:raise ValueError('invalid margin')
        self.enabled=force or bool(document.get('enabled',False))
        self.last_predictions={}
        self.last_action='skip'

    @classmethod
    def load(cls,path: str|Path|None=None,*,force=False):
        text=Path(path).read_text() if path else files('solvetide').joinpath('data/policy.json').read_text()
        return cls(json.loads(text),force=force)

    def plan(self,features,max_rank_budget):
        x=np.r_[1.,(vector(features)-self.mean)/self.scale]
        costs=np.exp(np.clip(self.weights@x,-30,20))
        self.last_predictions={a:float(c) for a,c in zip(ACTIONS,costs)}
        index=int(np.argmin(costs))
        self.last_action=ACTIONS[index] if self.enabled and costs[index]<costs[0]*(1-self.margin) else 'skip'
        return action_plan(self.last_action,max_rank_budget)


def fit(rows:list[dict],provenance:dict)->dict:
    X=np.vstack([vector(r['features']) for r in rows])
    mean,scale=X.mean(axis=0),X.std(axis=0)
    scale[scale<1e-8]=1.
    X=np.column_stack((np.ones(len(X)),(X-mean)/scale))
    penalty=np.eye(X.shape[1])*2
    penalty[0,0]=0
    weights=[]
    for action in ACTIONS:
        selection=np.array([r['action']==action for r in rows])
        if selection.sum()<10:raise ValueError('at least ten development pair records per action required')
        x=X[selection]
        # Failures remain explicit records and incur max-time penalty. They are
        # never silently discarded as attractive cheap interrupted actions.
        y=np.log([max(r['seconds'],30.0 if r['status']!='converged' else 1e-9) for r in rows if r['action']==action])
        weights.append(np.linalg.solve(x.T@x+penalty,x.T@y).tolist())
    return {'schema_version':2,'features':list(FEATURES),'actions':list(ACTIONS),'mean':mean.tolist(),'scale':scale.tolist(),
            'weights':weights,'refresh_margin':.05,'enabled':False,'training':provenance,'license':'MIT',
            'model':'per-joint-action ridge log complete-pair seconds',
            'action_semantics':'Quota capacity, not a claim that all slots contain valid vectors; actual used rank reported separately.'}
