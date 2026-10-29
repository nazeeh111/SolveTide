import importlib.util
import json
from pathlib import Path
from solvetide import Runtime,GradientResult

spec=importlib.util.spec_from_file_location('design_app_test',Path(__file__).parents[1]/'integration/design_app/design_app.py')
app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)


def test_failed_evaluation_returns_no_current_design(monkeypatch):
    class Refused(Runtime):
        def gradient(self,*args,**kwargs):
            return GradientResult('adjoint_failed',message='injected downstream solver refusal')
    monkeypatch.setattr(app,'Runtime',Refused)
    result=app.optimize(n=4)
    assert not result['success'] and result['status']=='solver_failed'
    assert result['parameters'] is result['objective'] is result['direct_checked_objective'] is None
    assert result['history'][-1]['gradient_norm'] is None
    json.dumps(result,allow_nan=False)


def test_disabled_model_uses_warm_without_forcing():
    result=app.optimize(n=8,strategy='learned')
    assert result['success'] and not result['forced_research_policy']
    assert result['effective_strategy']=='warm' and result['policy_gate']
    assert result['direct_checked_objective']<1e-8
