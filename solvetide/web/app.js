'use strict';
const $ = id => document.getElementById(id);
let current = null;
let running = false;
let requestID = 0;
let pendingCancel = Promise.resolve();
const controls = $('controls');
const canvas = $('field');
const ctx = canvas.getContext('2d');
const fmt = value => value == null || !Number.isFinite(value) ? '—' : Math.abs(value) < 0.001 || Math.abs(value) >= 10000 ? value.toExponential(2) : value.toPrecision(4);
function status(text, style='') { $('status').textContent=text; $('status').className='badge '+style; }
function unacceptedField(reason) {
  const image=$('empty').hidden?'The faded image is a previous field and is not accepted for the current inputs.':'No field is displayed.';
  canvas.setAttribute('aria-label',`${reason} No current accepted field. ${image}`);
}
function markStale() {
  unacceptedField('Inputs changed.');
  if (current) {
    current=null;
    status('Inputs changed · stale','stale');
    $('field-note').textContent='Displayed field belongs to previous inputs. Recompute to accept and export the changed design.';
    canvas.style.opacity='0.35';
  }
  $('export').disabled=true;
  $('material-value').textContent=Number($('material').value).toFixed(2);
  $('source-value').textContent=Number($('source').value).toFixed(2);
  $('contrast-value').textContent=Number($('contrast').value).toFixed(2);
  const elastic=$('family').value==='elasticity';
  $('field-title').textContent=elastic?'Displacement magnitude':'Temperature field';
  $('material-hint').textContent=elastic?'Positive stiffness multiplier':'Positive conductivity multiplier';
  $('field-unit').textContent=elastic?'Displacement · model units':'Temperature · model units';
  if (running) cancel(false);
}
controls.addEventListener('input',markStale);
controls.addEventListener('change',markStale);
async function post(path,data={}) {
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result=await response.json();
  if(!response.ok)throw new Error(result.error||'Request failed.');
  return result;
}
function payload(maxiter=2000) {
 return {family:$('family').value,n:Number($('n').value),boundary:$('boundary').value,
   parameters:{material:Number($('material').value),source:Number($('source').value),boundary_value:Number($('boundary-value').value),contrast:Number($('contrast').value)},strategy:$('strategy').value,maxiter};
}
async function compute(maxiter=2000) {
 const id=++requestID;
 running=true;current=null;$('export').disabled=true;$('error').hidden=true;
 $('solve').disabled=true;$('cancel').hidden=false;
 status('Assembling & solving…');
 unacceptedField('Computation pending.');
 $('field-note').textContent='Computing the current field and its implicit adjoint. Every residual uses the current equations.';
 canvas.style.opacity='0.35';
 try {
  await pendingCancel;
  if(id!==requestID)return;
  const result=await post('/api/solve',payload(maxiter));
  if(id!==requestID)return;
  updateInspection(result);
  if(!result.field) {
   unacceptedField(result.status==='cancelled'?'Computation cancelled.':'The new solve was rejected.');
   status(result.status==='cancelled'?'Cancelled':'Rejected · '+result.status,'failed');
   $('error').hidden=false;
   $('error').textContent=result.status==='cancelled'?'Computation cancelled. No new field or gradient was accepted.':'The solve did not meet the requested acceptance checks. No new field or gradient was returned. Recompute with the direct solver to try a fresh fallback.';
   $('field-note').textContent='No current accepted field. Any faded image is from an earlier solve.';
   return;
  }
  current=result;
  renderField(result);
  $('empty').hidden=true;canvas.style.opacity='1';$('export').disabled=false;
  status('Accepted · checked','accepted');
  $('field-note').textContent=`${result.input.family==='heat'?'Heat diffusion':'Linear elasticity'} · ${result.input.n} × ${result.input.n} divisions · field and gradient checked`;
 } catch(error) {
  if(id!==requestID)return;
  unacceptedField('Computation failed.');
  status('Computation failed','failed');$('error').textContent=error.message;$('error').hidden=false;
  $('field-note').textContent='No current result. Correct the inputs and compute again.';
 } finally {
  if(id===requestID){running=false;$('solve').disabled=false;$('cancel').hidden=true;}
 }
}
async function cancel(show=true) {
 unacceptedField('Computation cancelled.');
 ++requestID;running=false;$('solve').disabled=false;$('cancel').hidden=true;current=null;$('export').disabled=true;
 try{pendingCancel=post('/api/cancel');await pendingCancel;}catch(e){if(show){$('error').hidden=false;$('error').textContent=e.message;}}
 if(show){status('Cancelled');$('field-note').textContent='Computation cancelled. No new field or gradient was accepted.';}
}
function updateInspection(data) {
 const primal=data.primal||{},adjoint=data.adjoint||{};
 $('time').textContent=data.total_seconds==null?'—':(data.total_seconds*1000).toFixed(1)+' ms';
 $('iterations').textContent=(primal.iterations??'—')+' / '+(adjoint.iterations??'—');
 $('residual').textContent=fmt(primal.relative_residual);
 $('backward').textContent=fmt(primal.backward_error);
 $('memory').textContent=(primal.reused_rank??0)+' / '+((primal.memory_bytes??0)/1024).toFixed(1)+' KiB';
 $('objective').textContent=fmt(data.objective);
 $('cache').textContent=primal.cache_event||'No reusable state was accepted.';
 const policy=data.policy||{};
 $('policy-note').textContent=policy.note||'Policy state unavailable.';
 $('policy-state').replaceChildren();
 for(const [key,value] of [['requested_strategy',data.requested_strategy??data.input?.strategy],['effective_strategy',data.effective_strategy??data.input?.strategy],['policy_disabled',policy.policy_disabled??false],['policy_action',JSON.stringify(policy.policy_action||{})]]){
  const row=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=String(value);row.append(dt,dd);$('policy-state').append(row);
 }
 $('gradients').replaceChildren();
 const gradients=data.gradient||{'No accepted gradient':null};
 for(const [key,value] of Object.entries(gradients)) {
  const row=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');
  dt.textContent=key;dd.textContent=fmt(value);row.append(dt,dd);$('gradients').append(row);
 }
 $('details').textContent=JSON.stringify({identities:data.identities,primal,adjoint,assembly_seconds:data.assembly_seconds,runtime_state:data.runtime_state,policy:data.policy},null,2);
}
const colors=[[25,61,75],[28,129,116],[145,186,113],[242,220,112]];
function color(t){t=Math.max(0,Math.min(1,t));const x=t*3,i=Math.min(2,Math.floor(x)),r=x-i;return 'rgb('+colors[i].map((v,k)=>Math.round(v+(colors[i+1][k]-v)*r)).join(',')+')';}
function renderField(data) {
 const coords=data.coordinates,cells=data.cells,raw=data.field,elastic=data.input.family==='elasticity';
 const values=elastic?coords[0].map((_,i)=>Math.hypot(raw[2*i],raw[2*i+1])):raw;
 const low=Math.min(...values),high=Math.max(...values),span=high-low||1;
 $('min').textContent=fmt(low);$('max').textContent=fmt(high);
 canvas.setAttribute('aria-label',`${elastic?'Displacement magnitude':'Temperature'} field, ${data.input.n} divisions, range ${fmt(low)} to ${fmt(high)}. Accepted after residual and adjoint checks.`);
 const width=canvas.width,height=canvas.height,pad=65;
 ctx.clearRect(0,0,width,height);
 const size=Math.min(width-2*pad,height-2*pad),ox=(width-size)/2,oy=(height-size)/2;
 const sx=x=>ox+x*size,sy=y=>height-oy-y*size;
 for(let t=0;t<cells[0].length;t++) {
  const ids=[cells[0][t],cells[1][t],cells[2][t]];
  ctx.beginPath();ids.forEach((node,i)=>{const x=sx(coords[0][node]),y=sy(coords[1][node]);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.closePath();
  ctx.fillStyle=color((ids.reduce((sum,j)=>sum+values[j],0)/3-low)/span);ctx.fill();
  if(data.input.n<=20){ctx.strokeStyle='rgba(245,249,229,.15)';ctx.lineWidth=.5;ctx.stroke();}
 }
 ctx.strokeStyle='#285d55';ctx.lineWidth=1.2;ctx.strokeRect(ox,oy,size,size);
 if(elastic){
  const maximum=Math.max(high,1e-15),scale=0.06/maximum;
  ctx.strokeStyle='rgba(255,255,241,.9)';ctx.lineWidth=1.2;
  for(let i=0;i<coords[0].length;i+=Math.max(1,Math.floor(coords[0].length/65))){
   const x=sx(coords[0][i]),y=sy(coords[1][i]);const dx=raw[2*i]*size*scale,dy=-raw[2*i+1]*size*scale;
   ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+dx,y+dy);ctx.stroke();
  }
 }
}
controls.addEventListener('submit',event=>{event.preventDefault();compute();});
$('cancel').addEventListener('click',()=>cancel());
$('small-edit').addEventListener('click',()=>{$('contrast').value=Math.min(10,Number($('contrast').value)*1.05).toFixed(2);markStale();compute();});
$('invalidate').addEventListener('click',()=>{$('n').value=$('n').value==='20'?'32':'20';markStale();compute();});
$('force-failure').addEventListener('click',()=>{$('strategy').value='cold';markStale();compute(1);});
$('fallback').addEventListener('click',()=>{$('strategy').value='direct';markStale();compute();});
$('reset').addEventListener('click',async()=>{if(running)await cancel(false);try{await post('/api/reset');current=null;$('export').disabled=true;canvas.style.opacity='.35';unacceptedField('Solution memory cleared.');status('Memory cleared');$('cache').textContent='Solution memory cleared. The next computation starts without retained directions.';$('field-note').textContent='Memory cleared. Recompute to produce a new export.';}catch(e){$('error').hidden=false;$('error').textContent=e.message;}});
$('export').addEventListener('click',async()=>{
 if(!current)return;
 try{const res=await fetch('/api/export');const data=await res.json();if(!res.ok||data.revision!==current.revision)throw new Error('Result changed. Recompute before exporting.');
 const blob=new Blob([JSON.stringify(data,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`solvetide-${data.input.family}-revision-${data.revision}.json`;link.click();URL.revokeObjectURL(url);
 }catch(e){$('error').hidden=false;$('error').textContent=e.message;}
});
