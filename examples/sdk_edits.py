"""Run from any directory with SolveTide installed. No saved/private state."""
from solvetide import Runtime,SolverConfig,assemble
runtime=Runtime(SolverConfig(strategy='joint'))
for contrast,n in [(2.,16),(2.1,16),(2.1,20)]:
    system=assemble('heat',n,{'contrast':contrast})
    field=runtime.solve(system)
    sensitivity=runtime.gradient(system,field,target=.12)
    print(contrast,n,field.status,field.cache_event,field.reused_rank,sensitivity.gradient)
    if not sensitivity.accepted:
        raise RuntimeError('No usable field/gradient for this edit')
# Restart is explicit: no persistent solution cache is trusted after process exit.
runtime.reset()
assert runtime.state()['rank']==0
