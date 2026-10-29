"""Matched complete inverse-design runs through the separate client."""
import importlib.util
import numpy as np
from studies.run import write,ROOT
spec=importlib.util.spec_from_file_location('design_app',ROOT/'integration/design_app/design_app.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
runs=[]
for repeat in range(3):
    methods=['direct','warm','fixed','joint','learned','amg_reuse']
    np.random.default_rng(8200+repeat).shuffle(methods)
    for family in ('heat','elasticity'):
        for strategy in methods:
            r=module.optimize(family=family,n=24,strategy=strategy,force_policy=True)
            runs.append({'repeat':repeat,**r})
            print(f"{family} {strategy}: success={r['success']} J={r['direct_checked_objective']} t={r['seconds']:.3f}s",flush=True)
write(ROOT/'experiments/optimization.json',{'schema_version':1,'runs':runs,'source_sha256':{name:__import__('hashlib').sha256((ROOT/name).read_bytes()).hexdigest() for name in ['integration/design_app/design_app.py','studies/optimization.py','solvetide/runtime.py','solvetide/data/policy.json']},
      'limits':'Synthetic two-coefficient inverse problem. Identical starts/tolerances/iterations. Includes target generation, assembly, features, policy, basis, forward/adjoint and final direct verification.'})
