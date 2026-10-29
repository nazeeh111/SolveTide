from studies.run import trajectories,PROTOCOL,run_sequence
from solvetide import assemble
from solvetide.policy import ConstantPlan


def test_mesh_seed_and_boundaries_held_out():
    a,b=PROTOCOL['development'],PROTOCOL['confirmation']
    assert set(a['seeds']).isdisjoint(b['seeds'])
    assert set(a['meshes']).isdisjoint(b['meshes'])
    assert set(a['boundaries']).isdisjoint(b['boundaries'])
    cases=next(trajectories('confirmation'))[1]
    operators=[assemble(**{k:v for k,v in c.items() if k!='kind'}).identity['operator'] for c in cases[:4]]
    assert operators[0]==operators[1] and operators[2]==operators[3] and operators[0]!=operators[2]


def test_separate_and_balanced_use_real_pair_same_budget():
    cases=next(trajectories('development'))[1][:2]
    for method in ['separate','balanced']:
        run=run_sequence(cases,method)
        assert not run['failures'] and run['setup_seconds']>=0
        assert all(len(s['solves'])==2 for s in run['steps'])
        assert all(s['reference_errors']['gradient_relative_l2']<1e-6 for s in run['steps'])
