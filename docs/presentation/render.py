"""Plot actual editor exports. Existing NumPy/Matplotlib only; no solver runs.

Facet color is the mean of its three nodal values, matching the actual editor.
All mesh cells are retained. Difference is after minus before, in model units.
"""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as E
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm
from matplotlib.patches import Rectangle
HERE=Path(__file__).resolve().parent;a=json.loads((HERE/'editor-before.json').read_text());b=json.loads((HERE/'editor-after.json').read_text());assert a['input']['family']==b['input']['family']=='heat';assert a['coordinates']==b['coordinates'] and a['cells']==b['cells'];xy=np.array(a['coordinates']);tri=np.array(a['cells']).T;before=np.array(a['field']);after=np.array(b['field']);delta=after-before;assert len(before)==441 and len(tri)==800
values=[before,after,delta];low=float(min(before.min(),after.min()));high=float(max(before.max(),after.max()));limit=float(abs(delta).max());outputs={}
with mpl.rc_context({'font.family':['DejaVu Sans','Arial','sans-serif'],'font.size':17,'svg.fonttype':'none','svg.hashsalt':'solvetide-local-edit','path.simplify':False,'axes.spines.top':False,'axes.spines.right':False}):
 for mobile in (False,True):
  fig,axes=plt.subplots(3 if mobile else 1,1 if mobile else 3,figsize=(8,19) if mobile else (15,6),layout='constrained');fig.set_facecolor('#fffefa');fig.set_constrained_layout_pads(w_pad=.12,h_pad=.16,wspace=.06,hspace=.08)
  titles=['Before · contrast 2.00','After · contrast 2.10','Change · after − before']
  for i,ax in enumerate(axes):
   norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit) if i==2 else Normalize(vmin=low,vmax=high)
   image=ax.tripcolor(xy[0],xy[1],tri,facecolors=values[i][tri].mean(axis=1),norm=norm,cmap='RdBu_r' if i==2 else 'viridis',shading='flat',rasterized=False)
   ax.add_patch(Rectangle((.5,.5),.5,.5,fill=False,ls='--',lw=1.5,edgecolor='#151f23'))
   ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',xlabel='x · model coordinate',ylabel='y · model coordinate',title=titles[i]);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
   cb=fig.colorbar(image,ax=ax,orientation='horizontal',pad=.08,fraction=.07);cb.solids.set_rasterized(False);cb.set_label('Δ temperature · model units' if i==2 else 'Temperature · model units')
   if i==2:
    cb.set_ticks([-limit,0,limit]);cb.set_ticklabels([f'{-limit:.4f}','0',f'{limit:.4f}'])
   if mobile:ax.tick_params(labelsize=20);ax.title.set_size(23);ax.xaxis.label.set_size(20);ax.yaxis.label.set_size(20);cb.ax.tick_params(labelsize=18);cb.ax.xaxis.label.set_size(18)
  fig.suptitle('A local edit. A recomputed field.',fontsize=28,fontweight='bold')
  stem='field-edit-mobile' if mobile else 'field-edit';fig.savefig(HERE/(stem+'.png'),dpi=110,facecolor=fig.get_facecolor(),metadata={'Software':'SolveTide retained-editor figure'});fig.savefig(HERE/(stem+'.svg'),facecolor=fig.get_facecolor(),metadata={'Date':None,'Creator':'SolveTide retained-editor figure'});plt.close(fig)
  p=HERE/(stem+'.svg');E.register_namespace('','http://www.w3.org/2000/svg');E.register_namespace('xlink','http://www.w3.org/1999/xlink');root=E.fromstring(p.read_bytes());ns='{http://www.w3.org/2000/svg}';title=E.Element(ns+'title',{'id':stem+'-title'});title.text='SolveTide: local conductivity edit and global field response';desc=E.Element(ns+'desc',{'id':stem+'-desc'});desc.text='Actual accepted heat fields for contrasts 2.00 and 2.10 on the same unit-square mesh, with a separately scaled signed difference; the dashed top-right quadrant is the edited region. All 441 values and 800 triangular cells are retained.';root.insert(0,title);root.insert(1,desc);root.set('role','img');root.set('aria-labelledby',stem+'-title '+stem+'-desc')
  for node in root.iter():
   if node.tag in (ns+'text',ns+'tspan'):node.set('font-family','DejaVu Sans, Arial, Helvetica, sans-serif')
  p.write_bytes(E.tostring(root,encoding='utf-8',xml_declaration=True))
  outputs[stem]={ext:hashlib.sha256((HERE/(stem+'.'+ext)).read_bytes()).hexdigest() for ext in ['png','svg']}
source={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['editor-before.json','editor-after.json','editor-checks.json']}
manifest={'source_commit':'e639abebfbfef204d21a3af851fed5d6369f2ae3','source_tree':'3a8cdafc58a9ff9111e9bf5935bff5b4cadb75fe','inputs':source,'field_values_per_run':441,'triangles_per_panel':800,'shared_temperature_limits':[low,high],'symmetric_delta_limits':[-limit,limit],'delta_min_max':[float(delta.min()),float(delta.max())],'objective_before_after':[a['objective'],b['objective']],'contrast_gradient_before_after':[a['gradient']['contrast'],b['gradient']['contrast']],'transforms':['All source mesh cells; facet color is arithmetic mean of three stored nodal values, as in the editor','Delta computed as after minus before, before per-facet averaging','Same temperature limits for both fields; a separate symmetric signed difference scale','Dashed top-right quadrant identifies the edited material region; no geometry or coordinate changes','No solver, fitting, smoothing, normalization, interpolation or sample exclusion in presentation'],'limits':'Dimensionless supported discrete heat problem. Residual/gradient acceptance is not physical accuracy or a solution-error enclosure.','outputs':outputs};(HERE/'figure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
svgs=[(HERE/(n+'.svg')).read_text().split('?>',1)[-1] for n in ['field-edit','field-edit-mobile']];(HERE/'field-edit.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SolveTide field edit</title><style>body{margin:0;background:#fffefa}main{max-width:1200px;margin:auto}svg{display:block;width:100%;height:auto}.mobile{display:none}@media(max-width:640px){.wide{display:none}.mobile{display:block}}</style><main><div class="wide">'+svgs[0]+'</div><div class="mobile">'+svgs[1]+'</div></main></html>')
print(json.dumps(manifest,indent=2))
