/* Human-facing setup, bounded file browsing, and explicit result review. */
let workspaceView, workspaceTarget=null, workspaceRequest=0, pendingExistingRun=null;
const reviewCache=new Map(), reviewPending=new Set(), reviewPanels=new Map();
const hintKey='cluster-lab-hide-hints';
function displayValue(value){return value==null||value===''?'Not recorded':typeof value==='object'?JSON.stringify(value):String(value);}
function tableRows(rows){return '<table class="review-table"><tbody>'+rows.map(([key,value])=>'<tr><th>'+escapeHtml(key)+'</th><td>'+escapeHtml(displayValue(value))+'</td></tr>').join('')+'</tbody></table>';}
function setupCommand(){
 if($('setup-command-mode').value==='advanced')return JSON.parse($('run-command').value);
 const script=$('setup-script').value.trim();if(!script)throw Error('Choose an experiment script before saving.');
 return [$('setup-python').value.trim()||'{python}',script,...$('setup-args').value.split('\n').filter(x=>x.length)];
}
function commandMode(){const simple=$('setup-command-mode').value==='python';$('setup-python-fields').hidden=!simple;$('setup-command-advanced').hidden=simple;}
function parameterRow(key='',value=''){
 const row=document.createElement('div');row.className='parameter-row';
 row.innerHTML='<input aria-label="Parameter name" placeholder="batch_size"><input aria-label="Parameter value" placeholder="8"><button type="button" class="quiet-btn" aria-label="Remove parameter">Remove</button>';
 row.children[0].value=key;row.children[1].value=typeof value==='string'?value:JSON.stringify(value);row.children[2].onclick=()=>row.remove();$('setup-parameters').append(row);
}
function continuationFields(){
 const resumable=$('run-resume').checked;
 $('run-auto').disabled=!resumable;if(!resumable)$('run-auto').checked=false;
 for(const id of ['run-tasks','run-workers','run-concurrency']){$(id).disabled=!resumable;if(!resumable)$(id).value=1;}
 $('setup-continuation-help').textContent=resumable?'Your adapter must save and validate checkpoints and report completion. Automatic continuation can submit another allocation after useful progress.':'Single allocation: run one command once. No automatic restart or sharding. A successful exit confirms command completion; scientific results come from your script.';
}
function fillSetupFields(cfg){
 const command=cfg.command||[],simple=command.length>=2&&command[1].endsWith('.py')&&!command.slice(2).some(x=>x.includes('\n'));
 $('setup-command-mode').value=simple?'python':'advanced';$('setup-python').value=simple&&!command[0].startsWith('/absolute/path/')?command[0]:currentProfile()?.python||'{python}';$('setup-script').value=simple&&!command[1].startsWith('/absolute/path/')?command[1]:'';$('setup-args').value=simple?command.slice(2).join('\n'):'';commandMode();
 $('setup-notes').value=cfg.notes||'';$('setup-model').value=cfg.recipe?.model_path||'';$('setup-data').value=cfg.recipe?.data_path||'';
 for(const key of ['model','data','environment'])$('setup-'+key+'-version').value=cfg.fingerprints?.[key]||'';
 $('setup-parameters').replaceChildren();for(const [key,value]of Object.entries(cfg.recipe||{}))if(!['model_path','data_path'].includes(key))parameterRow(key,value);
 continuationFields();syncHints();
}
function setupConfiguration(cfg){
 const recipe=Object.create(null);
 for(const row of $('setup-parameters').children){const key=row.children[0].value.trim(),raw=row.children[1].value;if(!key)throw Error('Each parameter needs a name.');if(Object.hasOwn(recipe,key))throw Error('Duplicate parameter: '+key);try{recipe[key]=JSON.parse(raw);}catch{recipe[key]=raw;}}
 for(const [field,key]of [['setup-model','model_path'],['setup-data','data_path']])if($(field).value.trim())recipe[key]=$(field).value.trim();
 const fingerprints={...cfg.fingerprints};for(const key of ['model','data','environment']){const value=$('setup-'+key+'-version').value.trim();if(value)fingerprints[key]=value;else delete fingerprints[key];}
 return {...cfg,recipe,fingerprints,notes:$('setup-notes').value};
}
async function openWorkspace(target=null,path=''){
 if(!profileId){notice('Choose a cluster profile first.');return;}
 workspaceTarget=target;workspaceView=null;$('workspace-title').textContent=target?'Choose a cluster path':'Explore cluster files';$('workspace-filter').value='';
 if(!$('workspace-dialog').open)$('workspace-dialog').showModal();await browseWorkspace(path);
}
async function browseWorkspace(path='',offset=0){
 if(workspaceView&&path!==workspaceView.path&&path!==workspaceView.absolute_path)$('workspace-filter').value='';
 const request=++workspaceRequest,profile=profileId;$('workspace-status').textContent='Reading cluster workspace…';$('workspace-content').replaceChildren();$('workspace-use').hidden=true;$('workspace-import').hidden=true;$('workspace-more').hidden=true;
 try{
  const result=(await api('browse',{profile,path,search:$('workspace-filter').value,offset,hidden:$('workspace-hidden').checked})).result;
  if(request!==workspaceRequest||profile!==profileId)return;workspaceView=result;$('workspace-location').value=result.absolute_path;$('workspace-up').disabled=result.parent==null;
  $('workspace-status').textContent=result.scan_limited?'Showing entries from a bounded scan of the first 2,000 directory entries. Open a specific path for larger folders.':result.truncated?'Preview: first 64 KiB. The full file stays on the cluster.':result.kind==='directory'?'Read-only folder · '+result.entries.length+' entries on this page':'Read-only file · '+result.bytes.toLocaleString()+' bytes';
  $('workspace-use').hidden=!workspaceTarget;
  if(result.kind==='directory'){
   $('workspace-content').innerHTML='<div class="file-list">'+result.entries.map((entry,i)=>'<div class="file-row"><button class="file-name" data-file="'+i+'" '+(['link','special'].includes(entry.kind)?'disabled':'')+'><span>'+({directory:'▤',file:'▧',link:'↗',special:'·'}[entry.kind])+'</span>'+escapeHtml(entry.name)+'</button><span class="muted">'+escapeHtml(entry.kind)+(entry.kind==='file'?' · '+entry.bytes.toLocaleString()+' B':'')+'</span>'+(workspaceTarget?'<button class="quiet-btn" data-use="'+i+'">Select</button>':'')+'</div>').join('')+'</div>';
   document.querySelectorAll('[data-file]').forEach(el=>el.onclick=()=>browseWorkspace(result.entries[Number(el.dataset.file)].path));
   document.querySelectorAll('[data-use]').forEach(el=>el.onclick=()=>useWorkspacePath(result.root+'/'+result.entries[Number(el.dataset.use)].path));
   $('workspace-more').hidden=!result.more;
  }else{
   const pre=document.createElement('pre');pre.className='file-preview';pre.textContent=result.preview_available?result.text:result.reason;$('workspace-content').append(pre);
   $('workspace-import').hidden=!result.preview_available||result.truncated||!/[.](json|ya?ml)$/i.test(result.path);
  }
 }catch(error){if(request===workspaceRequest)$('workspace-status').textContent=error.message;}
}
function useWorkspacePath(path){if(!workspaceTarget)return;$(workspaceTarget).value=path;$('workspace-dialog').close();}
async function loadReview(run){
 const key=profileId+':'+run.id,cached=reviewCache.get(key);if(reviewFresh(cached,run))return cached.data;
 const result=(await api('review',{profile:profileId,id:run.id})).result;reviewCache.set(key,{time:Date.now(),state:run.state,data:result});return result;
}
function reviewFresh(record,run){return record&&record.state===run.state&&Date.now()-record.time<(run.profile.simulation?2000:30000);}
function renderRunReview(run){
 let section=$('run-review');if(!section){section=document.createElement('section');section.id='run-review';$('detail').querySelector('.detail-body').prepend(section);}
 const worker=Number($('detail-worker')?.value||0),key=profileId+':'+run.id,record=reviewCache.get(key),workerReview=record?.data.workers.find(w=>w.index===worker),metrics=workerReview?.metrics||{};
 const summary=[['Completion',run.config.completion],['Script / command',run.config.command.join(' ')],['Model / checkpoint',run.config.recipe.model_path],['Dataset',run.config.recipe.data_path],['Parameters',run.config.recipe],['Fingerprints',run.config.fingerprints],['Output folder',run.profile.remote_root+'/cluster-ui/'+run.id],['Slurm jobs',run.waves.map(w=>w.job_id||'Submission unconfirmed').join(', ')],['Requested resources','1 node per worker · '+run.config.gpus+' GPU · '+(run.config.cpus||'cluster-default')+' CPUs'],['Work assignment',run.config.workers+' fixed logical worker(s); '+run.control.concurrency+' concurrent'],['Optional budget',Object.keys(run.control.budget).length?run.control.budget:'No overall budget configured']];
 section.innerHTML='<h3>Experiment review</h3><p class="form-help">'+escapeHtml(run.config.resume_capable?'Progress and completion are declared by the checkpoint adapter.':'This is a single command run. Process success alone does not establish model quality.')+'</p><details open><summary>Reported results · worker '+worker+'</summary>'+ (Object.keys(metrics).length?tableRows(Object.entries(metrics)):'<p class="form-help">'+(record?'No numeric metrics reported in the recent journal. Inspect saved result files below.':'Loading recent reported metrics…')+'</p>')+'</details><details><summary>Recipe, inputs, and resources</summary>'+tableRows(summary)+'</details><div class="detail-actions"><button class="secondary" id="review-files">Open run files</button><button class="secondary" id="review-compare">Compare with another run</button></div>';
 $('review-files').onclick=()=>openWorkspace(null,'cluster-ui/'+run.id);$('review-compare').onclick=()=>openComparison(run.id);
 // Polling must not undo the reader's choice to expand or close a section.
 const panels=reviewPanels.get(key)||[true,false];
 section.querySelectorAll('details').forEach((panel,index)=>{panel.open=panels[index];panel.ontoggle=()=>{if(panel.isConnected){panels[index]=panel.open;reviewPanels.set(key,panels);}};});
 if(workerReview?.error||record?.data.workers_limited)section.append(Object.assign(document.createElement('p'),{className:'form-help',textContent:workerReview?.error||'This summary reads the first 16 workers. Use worker logs or saved run files for the remaining workers.'}));
 const configurationButton=$('run-config-action');if(configurationButton)configurationButton.textContent='Duplicate setup';
 if(!reviewFresh(record,run)&&!reviewPending.has(key)){
  reviewPending.add(key);loadReview(run).then(()=>{if(selectedId===run.id&&profileId+':'+run.id===key)renderRunReview(run);}).catch(error=>{if(selectedId===run.id)$('run-review').append(Object.assign(document.createElement('p'),{className:'error',textContent:error.message}));}).finally(()=>reviewPending.delete(key));
 }
}
function openComparison(id=selectedId){
 const options=runs.map(r=>'<option value="'+escapeHtml(r.id)+'">'+escapeHtml(r.config.name)+'</option>').join('');$('compare-left').innerHTML=options;$('compare-right').innerHTML=options;
 if(id&&runs.some(r=>r.id===id))$('compare-left').value=id;const other=runs.find(r=>r.id!==$('compare-left').value);if(other)$('compare-right').value=other.id;
 $('comparison-content').textContent=runs.length<2?'Create or duplicate another experiment to compare it here.':'';$('compare-load').disabled=runs.length<2;$('compare-dialog').showModal();
}
async function compareRuns(){
 const left=runs.find(r=>r.id===$('compare-left').value),right=runs.find(r=>r.id===$('compare-right').value);if(!left||!right)return;
 if(left.id===right.id){$('comparison-content').textContent='Choose two different experiments.';return;}
 $('compare-load').disabled=true;$('comparison-content').textContent='Reading recent results from the cluster…';
 try{
  const reviews=await Promise.all([loadReview(left),loadReview(right)]),lm=reviews[0].workers[0]?.metrics||{},rm=reviews[1].workers[0]?.metrics||{};
  const rows=[['Status',left.state,right.state],['Committed items',completed(left)+' / '+left.config.task_count,completed(right)+' / '+right.config.task_count],['Identity',left.identity,right.identity],['Model revision',left.config.fingerprints.model,right.config.fingerprints.model],['Data revision',left.config.fingerprints.data,right.config.fingerprints.data],['Environment revision',left.config.fingerprints.environment,right.config.fingerprints.environment],['Completion condition',left.config.completion,right.config.completion],['Parameters',left.config.recipe,right.config.recipe],['Command',left.config.command,right.config.command],['GPUs per worker',left.config.gpus,right.config.gpus],['Allocations',left.waves.length,right.waves.length],...Array.from(new Set([...Object.keys(lm),...Object.keys(rm)])).sort().map(key=>['Worker 0 · '+key,lm[key],rm[key]])];
  $('comparison-content').innerHTML='<p class="form-help">Highlighted rows differ. Metrics below are the latest reported values for worker 0, not averages across workers. Inspect each run for other workers.</p><div class="table-scroll"><table class="review-table comparison-table"><thead><tr><th>Setting or result</th><th>'+escapeHtml(left.config.name)+'</th><th>'+escapeHtml(right.config.name)+'</th></tr></thead><tbody>'+rows.map(([key,a,b])=>'<tr class="'+(JSON.stringify(a)!==JSON.stringify(b)?'different':'')+'"><th>'+escapeHtml(key)+'</th><td>'+escapeHtml(displayValue(a))+'</td><td>'+escapeHtml(displayValue(b))+'</td></tr>').join('')+'</tbody></table></div>';
 }catch(error){$('comparison-content').textContent=error.message;}finally{$('compare-load').disabled=false;}
}
async function reviewLaunch(existingRun=null){
 try{
  const cfg=existingRun?existingRun.config:(await api('validate',{yaml:configSource()})).config,p=existingRun?existingRun.profile:currentProfile();pendingExistingRun=existingRun;
  $('launch-summary').innerHTML=tableRows([['Experiment',cfg.name],['Cluster',p.name],['Workspace',p.remote_root],['Command',cfg.command],['Inputs / parameters',cfg.recipe],['Resources','1 node per worker · '+cfg.gpus+' GPU(s) · '+(cfg.cpus||'cluster-default')+' CPU(s)'],['Logical workers / concurrency',cfg.workers+' / '+cfg.concurrency],['Possible partitions / requested wall time',p.partitions.map(x=>x.name+' · '+x.time_limit).join('; ')],['Continuation',cfg.auto_continue?'Automatic after a validated checkpoint and useful progress':'No automatic successor'],['Overall budget',Object.keys(cfg.budget).length?cfg.budget:'None configured'],['Completion',cfg.completion]])+'<p class="form-help">Submission refreshes cluster limits and queue estimates. The chosen partition and Slurm job ID are recorded in this experiment. Stop run disables future handoffs and cancels its recorded job.</p>';
  $('launch-review-dialog').showModal();
 }catch(error){$('run-error').textContent=error.message;}
}
function syncHints(){const hidden=localStorage.getItem(hintKey)==='yes';document.querySelectorAll('.dismissible-hint').forEach(el=>el.hidden=hidden);if($('guide-show-hints'))$('guide-show-hints').checked=!hidden;}
function openGuide(topic=''){if(!$('guide-dialog').open)$('guide-dialog').showModal();$('guide-search').value=topic;filterGuide();}
function filterGuide(){const query=$('guide-search').value.toLowerCase();document.querySelectorAll('#guide-topics details').forEach(el=>{el.hidden=!el.textContent.toLowerCase().includes(query);if(query&&!el.hidden)el.open=true;});}
$('setup-command-mode').onchange=commandMode;$('run-resume').onchange=continuationFields;$('add-parameter').onclick=()=>parameterRow();
document.querySelectorAll('[data-pick]').forEach(button=>button.onclick=()=>openWorkspace(button.dataset.pick));
for(const id of ['workspace-open','guide-files'])$(id).onclick=()=>openWorkspace();
for(const id of ['compare-open','guide-compare'])$(id).onclick=()=>openComparison();$('guide-new').onclick=openRun;
$('workspace-location-form').onsubmit=e=>{e.preventDefault();browseWorkspace($('workspace-location').value);};
$('workspace-filter-apply').onclick=()=>browseWorkspace(workspaceView?.path||'');$('workspace-up').onclick=()=>browseWorkspace(workspaceView?.parent||'');$('workspace-more').onclick=()=>browseWorkspace(workspaceView.path,workspaceView.offset+200);$('workspace-use').onclick=()=>useWorkspacePath(workspaceView.absolute_path);
$('workspace-import').onclick=async()=>{try{const parsed=await api('validate',{yaml:workspaceView.text});draft=parsed.config;draftId=crypto.randomUUID().replaceAll('-','');fillForm();$('run-yaml').value=parsed.yaml;mode='yaml';$('guided-fields').hidden=true;$('yaml-fields').hidden=false;$('guided-tab').classList.remove('active');$('yaml-tab').classList.add('active');$('workspace-dialog').close();if(!$('run-dialog').open)$('run-dialog').showModal();}catch(error){$('workspace-status').textContent='This file is not a dashboard run configuration: '+error.message;}};
$('compare-load').onclick=compareRuns;$('run-form').onsubmit=e=>{e.preventDefault();reviewLaunch();};$('launch-confirm').onclick=()=>{$('launch-review-dialog').close();if(pendingExistingRun){selectedId=pendingExistingRun.id;pendingExistingRun=null;runAction('launch');}else saveRun(true);};
$('help-open').onclick=()=>openGuide();document.querySelectorAll('[data-help]').forEach(el=>el.onclick=()=>openGuide(el.dataset.help));
document.querySelectorAll('[data-dismiss-hint]').forEach(el=>el.onclick=()=>{localStorage.setItem(hintKey,'yes');syncHints();});$('guide-show-hints').onchange=()=>{localStorage.setItem(hintKey,$('guide-show-hints').checked?'no':'yes');syncHints();};$('guide-search').oninput=filterGuide;syncHints();
