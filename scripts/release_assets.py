"""Package permitted evidence and hash final release bytes. Does not publish."""
from pathlib import Path
from hashlib import sha256
import json
import shutil
import zipfile
ROOT=Path(__file__).resolve().parents[1]
dist=ROOT/'dist';dist.mkdir(exist_ok=True)
with zipfile.ZipFile(dist/'solvetide-0.1.0-examples.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in (ROOT/'examples/results-v0.1.0').glob('*.json'):z.write(p,p.name)
    z.write(ROOT/'solvetide/checker.py','checker.py')
    z.writestr('README.txt','SolveTide0.1.0 newly computed original-equation bundles.\nRun: python3 -I -S checker.py heat-contrast-2.1.json\nThe checker needs only Python standard library.\nEach accepted bundle includes original sparse equations, adjoint and derivative data.\nrejected-one-iteration.json is a failure report, not a checkable accepted field.\nNo physical-model or solution-error enclosure is certified.\n')
with zipfile.ZipFile(dist/'solvetide-0.1.0-study.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['confirmation.json','optimization.json','memory.json','numerical-validation.json','training.json','installed-check.json','editor-latency.json']:
        p=ROOT/'experiments'/name
        if not p.exists():raise RuntimeError(f'missing executed release gate: {name}')
        z.write(p,'experiments/'+name)
    for name in ['studies/protocol.json','solvetide/data/policy.json','docs/measurements.md','docs/benchmarks.png','docs/architecture.mmd']:
        z.write(ROOT/name,name)
    z.writestr('README.txt','SolveTide0.1.0 finite CPU research evidence.\nModel/data are original synthetic trajectories.\nLearning is controlled by development gate; forced learned confirmation rows are not deployed defaults.\nRaw statuses, costs, source/protocol/model hashes and failures are retained.\nSee docs/measurements.md and source repository for reproduction/limits.\n')
if (ROOT/'docs/demo.mp4').exists():shutil.copyfile(ROOT/'docs/demo.mp4',dist/'solvetide-0.1.0-demo.mp4')
shutil.copyfile(ROOT/'solvetide/checker.py',dist/'checker.py')
assets=sorted(p for p in dist.iterdir() if p.is_file() and p.name!='SHA256SUMS')
(dist/'SHA256SUMS').write_text(''.join(f'{sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in assets))
print(json.dumps({p.name:p.stat().st_size for p in assets},indent=2))
