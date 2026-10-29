import json
from pathlib import Path
import numpy as np
import pytest
from solvetide.policy import FEATURES,ACTIONS,RankPolicy,fit,action_plan


def test_plans_allocate_equal_budget():
    for n in range(33):
        for action in ACTIONS:
            p=action_plan(action,n)
            assert p['primal_quota']+p['adjoint_quota']==n
            assert p['primal_rank']==p['primal_quota']


def test_fitted_policy_no_confirmation():
    path=Path(__file__).parents[1]/'solvetide/data/policy.json'
    if not path.exists() or json.loads(path.read_text())['schema_version']!=2:pytest.skip('pair training underway')
    d=json.loads(path.read_text())
    assert all(n.startswith('development-') and not n.endswith('104') for n in d['training']['trajectory_names'])
    p=RankPolicy(d)
    plan=p.plan(dict(zip(FEATURES,np.ones(9))),12)
    assert plan['primal_quota']+plan['adjoint_quota']==12
    if not d['enabled']:assert plan['refresh'] is False


def test_invalid_weights_rejected():
    rows=[{'features':dict(zip(FEATURES,np.ones(9))),'action':a,'seconds':.01,'status':'converged'} for a in ACTIONS for _ in range(11)]
    d=fit(rows,{})
    d['weights'][0][0]=float('nan')
    with pytest.raises(ValueError):RankPolicy(d)
