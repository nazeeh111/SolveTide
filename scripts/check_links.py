"""Check local Markdown references without treating future public URLs as proof."""
from pathlib import Path
import re
root=Path(__file__).resolve().parents[1]
errors=[]
for path in [root/'README.md',*root.joinpath('docs').rglob('*.md')]:
    for match in re.finditer(r'\]\(([^)]+)\)',path.read_text()):
        target=match.group(1).split('#',1)[0]
        if not target or '://' in target or target.startswith('mailto:'):continue
        if not (path.parent/target).exists():errors.append(f'{path.relative_to(root)}: {target}')
if errors:raise SystemExit('\n'.join(errors))
print('Local documentation links resolve.')
