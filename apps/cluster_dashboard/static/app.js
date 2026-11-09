/* No framework or external resources. The server is the source of truth. */
const $ = id => document.getElementById(id);
let boot, profileId, runs = [], selectedId, draft, mode = "guided", polling = false, draftId;
const noteDrafts = {};
let stream = {key:"", offsets:{}, buffer:{}, metrics:[], log:""};
const escapeHtml = x => String(x ?? "").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = x => x == null ? "Unknown" : x < 60 ? Math.round(x)+"s" : x < 3600 ? Math.round(x/60)+"m" : (x/3600).toFixed(1)+"h";
function notice(message=""){ $("notice").textContent=message; $("notice").hidden=!message; }
async function api(path, body){
  const response=await fetch("/api/"+path,{method:body?"POST":"GET",headers:body?{"Content-Type":"application/json","X-Dashboard-Token":boot.token}:{},body:body?JSON.stringify(body):undefined});
  const data=await response.json(); if(!response.ok)throw Error(data.error || response.statusText); return data;
}
function currentProfile(){return boot.profiles.find(p=>p.id===profileId)?.profile;}
function completed(run){return (run.workers||[]).reduce((n,w)=>n+(Number.isFinite(w.result?.completed)&&w.result.completed>=0?w.result.completed:0),0);}
async function bootstrap(){
  boot=await api("bootstrap"); const previous=profileId || localStorage.getItem("cluster-profile");
  profileId=boot.profiles.some(p=>p.id===previous)?previous:boot.profiles[0]?.id;
  $("profile-select").innerHTML=boot.profiles.map(p=>'<option value="'+escapeHtml(p.id)+'">'+escapeHtml(p.profile.name)+'</option>').join("");
  if(!profileId)$("profile-select").innerHTML='<option>No profiles yet</option>';
  $("profile-select").value=profileId||""; renderProfile(); await refresh();
}
function renderProfile(){
 const p=currentProfile(); $("profile-description").textContent=p?(p.simulation?"A CPU-only sandbox for the full experiment lifecycle.":p.ssh_host+" · "+p.remote_root):"Add your SSH destination and cluster settings.";
}
async function refresh(force=false){
 if(polling||!profileId)return; polling=true; const requestedProfile=profileId;
 try{
  const data=force?await api("refresh",{profile:profileId}):await api("runs?profile="+encodeURIComponent(profileId));
  if(requestedProfile!==profileId)return;
  runs=data.runs; $("connection-label").textContent=data.offline?"Offline · saved records":currentProfile().simulation?"Simulator connected":"Cluster connected";
  $("connection-dot").classList.toggle("offline",!!data.offline);
  if(data.offline)notice(data.error); else notice();
  $("last-sync").textContent="Synced "+new Date().toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"});
  renderRuns(); renderDetail();
  if(!data.offline)await refreshStreams();
 }catch(e){notice(e.message);$("connection-label").textContent="Connection needs attention";$("connection-dot").classList.add("offline");}
 finally{polling=false;if(requestedProfile!==profileId)refresh();}
}
function renderRuns(){
 $("run-count").textContent=runs.length; $("empty").hidden=!!runs.length;
 $("stat-active").textContent=runs.filter(r=>["queued","submitting","running"].includes(r.state)).length;
 $("stat-complete").textContent=runs.filter(r=>r.state==="complete").length;
 $("stat-waves").textContent=runs.reduce((n,r)=>n+r.waves.length,0);
 $("stat-hours").textContent=runs.reduce((n,r)=>n+r.waves.filter(w=>w.status==="submitted").reduce((a,w)=>a+w.reserved_gpu_hours,0),0).toFixed(1);
 $("run-list").innerHTML=runs.map(r=>{const n=completed(r),pc=Math.min(100,100*n/r.config.task_count);return '<button class="run-row '+(r.id===selectedId?"chosen":"")+'" data-run="'+escapeHtml(r.id)+'"><div class="row-top"><span class="run-name">'+escapeHtml(r.config.name)+'</span><span class="badge '+escapeHtml(r.state)+'">'+escapeHtml(r.state.replaceAll("_"," "))+'</span></div><div class="run-meta"><span>'+escapeHtml(r.config.kind)+' · '+r.config.gpus+' GPU'+(r.config.gpus!==1?"s":"")+' / worker</span><span>'+n+' / '+r.config.task_count+' items</span></div><div class="track"><div class="track-fill" ></div></div></button>';}).join("");
 // CSP forbids inline styles; set progress geometry through the DOM.
 document.querySelectorAll(".track-fill").forEach((el,i)=>el.style.width=Math.min(100,100*completed(runs[i])/runs[i].config.task_count)+"%");
 document.querySelectorAll("[data-run]").forEach(el=>el.onclick=()=>{selectedId=el.dataset.run;stream={key:"",offsets:{},buffer:{},metrics:[],log:""};renderRuns();renderDetail(true);refreshStreams().catch(e=>notice(e.message));});
}
function renderDetail(force=false){
 const r=runs.find(r=>r.id===selectedId);if(!r)return;
 const worker=$("detail-worker")?.value||"0", viewedWave=$("detail-wave")?.value||"latest";
 const focus=document.activeElement; const editing=focus&&["run-notes","detail-concurrency"].includes(focus.id);
 if(editing&&!force)return;
 const wave=r.waves.at(-1), p=wave?.partition, reason=r.attention||r.error||wave?.error||"";
 const previousLog=stream.log;
 $("detail").innerHTML='<div class="detail-top"><span class="badge '+escapeHtml(r.state)+'">'+escapeHtml(r.state.replaceAll("_"," "))+'</span><h2>'+escapeHtml(r.config.name)+'</h2><span class="detail-id">'+escapeHtml(r.id)+' · '+r.identity.slice(0,12)+'</span><div class="detail-actions"><button class="secondary" id="run-resume-action">'+(r.state==="draft"?"Launch":"Resume")+'</button><button class="secondary danger" id="run-stop-action">Stop run</button><button class="secondary" id="run-config-action">View configuration</button></div></div><div class="detail-body"><div class="key-values"><div><span>COMMITTED ITEMS</span><strong>'+completed(r)+' / '+r.config.task_count+'</strong></div><div><span>PARTITION</span><strong>'+escapeHtml(p?.name||"Not submitted")+'</strong></div><div><span>QUEUE / STARTUP ESTIMATE</span><strong>'+fmt(p?.queue_seconds)+' / '+fmt(p?.startup_seconds)+'</strong></div><div><span>FIRST RESULT / FINISH ESTIMATE</span><strong>'+fmt(p?.first_result_seconds)+' / '+fmt(p?.finish_seconds)+'</strong></div><div><span>CONTINUATION</span><strong>'+(r.control.auto_continue?"Automatic":"Manual")+'</strong></div><div><span>WORKER CONCURRENCY</span><input id="detail-concurrency" aria-label="Concurrent workers" type="number" min="1" max="'+r.config.workers+'" value="'+escapeHtml(r.control.concurrency)+'"><button id="apply-concurrency" class="quiet-btn">Apply</button></div></div><p class="form-help">'+escapeHtml(r.scheduler?.state||"Ready")+' · '+escapeHtml(r.scheduler?.reason||"Experiment records are durable on the cluster.")+'</p>'+(reason?'<div class="detail-warning">'+escapeHtml(reason)+'</div>':"")+'<div class="subheading"><h3>WORKER METRICS</h3><select id="detail-worker" aria-label="Worker">'+Array.from({length:r.config.workers},(_,i)=>'<option value="'+i+'">Worker '+i+'</option>').join("")+'</select></div><svg id="loss-chart" viewBox="0 0 400 95" preserveAspectRatio="none" role="img" aria-label="Observed loss over time"></svg><div id="metric-caption">Metrics appear when the adapter reports them.</div><div id="gpu-caption" class="muted"></div><div class="subheading"><h3>ALLOCATION LOG</h3><span class="muted">Incremental updates</span></div><pre id="run-log" class="logs"></pre><div class="subheading"><h3>CHECKPOINTS & NOTES</h3><button id="save-notes" class="quiet-btn">Save notes</button></div><p class="muted">'+r.workers.filter(w=>w.checkpoint||w.result?.checkpoint_files&&Object.keys(w.result.checkpoint_files).length).length+' worker(s) with saved checkpoint evidence</p><textarea id="run-notes" class="notes" aria-label="Run notes"></textarea></div>';
 $("run-notes").value=noteDrafts[r.id]??r.notes;$("run-notes").oninput=()=>{noteDrafts[r.id]=$("run-notes").value;};
 const logHeading=[...document.querySelectorAll(".subheading")].find(el=>el.textContent.includes("ALLOCATION LOG"));
 const waveSelect=document.createElement("select");waveSelect.id="detail-wave";waveSelect.setAttribute("aria-label","Log allocation");
 waveSelect.innerHTML='<option value="latest">Latest allocation</option>'+r.waves.map(w=>'<option value="'+escapeHtml(w.number)+'">Allocation '+escapeHtml(Number(w.number)+1)+'</option>').join("");waveSelect.value=viewedWave;waveSelect.onchange=()=>{stream={key:"",offsets:{},buffer:{},metrics:[],log:""};$("run-log").textContent="Loading selected allocation…";refreshStreams().catch(e=>notice(e.message));};logHeading?.append(waveSelect);$("detail-worker").value=worker;$("run-log").textContent=previousLog||"Waiting for allocation output…";
 $("run-resume-action").disabled=r.state==="complete"||["queued","submitting","running"].includes(r.state)&&!r.attention;
 $("run-stop-action").disabled=["complete","cancelled","draft"].includes(r.state);
 $("run-resume-action").onclick=()=>r.state==="draft"?reviewLaunch(r):runAction("resume");
 $("run-stop-action").onclick=()=>runAction("stop");
 $("apply-concurrency").onclick=()=>runAction("concurrency",{concurrency:Number($("detail-concurrency").value)});
 $("save-notes").onclick=async()=>{await runAction("notes",{notes:$("run-notes").value});};
 $("run-config-action").onclick=()=>{draftId=crypto.randomUUID().replaceAll("-","");draft=structuredClone(r.config);fillForm();mode="guided";setMode("yaml");$("run-dialog").showModal();};
 $("detail-worker").onchange=()=>{renderRunReview(r);stream={key:"",offsets:{},buffer:{},metrics:[],log:""};refreshStreams().catch(e=>notice(e.message));};
 drawMetrics();renderRunReview(r);
}
async function runAction(action,extra={}){
 const messages={launch:'Checking cluster limits and submitting…',resume:'Reconciling previous allocation and resuming…',stop:'Sending stop and cancellation request…',concurrency:'Updating worker concurrency…',notes:'Saving notes…'};
 $('action-status').hidden=false;$('action-status').textContent=messages[action]||'Working…';
 const controls=$('detail').querySelectorAll('button');controls.forEach(b=>b.disabled=true);
 try{await api(action,{profile:profileId,id:selectedId,...extra});await refresh(true);$('action-status').textContent='Request finished. The recorded run state is shown below.';}catch(e){notice(e.message);$('action-status').textContent='Request needs attention. Inspect the message before retrying.';}finally{controls.forEach(b=>b.disabled=false);renderDetail(true);}
}
async function refreshStreams(){
 const r=runs.find(r=>r.id===selectedId);if(!r||!r.waves.length||!$("detail-worker"))return;
 const i=$("detail-worker").value, waveValue=$("detail-wave")?.value||"latest",wave=waveValue==="latest"?r.waves.at(-1):r.waves.find(w=>w.number===Number(waveValue)),key=r.id+":"+i;
 if(stream.key!==key)stream={key,offsets:{},buffer:{},metrics:[],log:""};
 const target=stream, profile=profileId;
 if(target.pending)return;
 target.pending=true;
 try{
 const logFile="waves/"+wave.number+"/worker-"+i+".log";
 if(stream.logKey!==logFile){stream.log="";stream.offsets[logFile]=0;stream.logKey=logFile;}
 for(const file of ["workers/"+i+"/metrics.jsonl",logFile]){
  const data=await api("tail?"+new URLSearchParams({profile,id:r.id,file,offset:target.offsets[file]||0}));
  // A delayed SSH response must never appear under a different run or worker.
  if(stream!==target||profileId!==profile||selectedId!==r.id||$("detail-worker")?.value!==i||($("detail-wave")?.value||"latest")!==waveValue)return;
  stream.offsets[file]=data.offset;
  if(file.endsWith(".jsonl")){
   const lines=((stream.buffer[file]||"")+data.text).split("\n");stream.buffer[file]=lines.pop();
   for(const line of lines){try{const row=JSON.parse(line);if(row&&typeof row==="object"&&!Array.isArray(row))stream.metrics.push(row);}catch{}}
   stream.metrics=stream.metrics.slice(-200);
  }else stream.log=(stream.log+data.text).slice(-20000);
 }
 if($("run-log"))$("run-log").textContent=stream.log||"Waiting for allocation output…";drawMetrics();
 }finally{target.pending=false;}
}
function drawMetrics(){
 const svg=$("loss-chart");if(!svg)return;svg.replaceChildren();
 const losses=stream.metrics.filter(m=>Number.isFinite(m.loss)||Number.isFinite(m.train_loss)).map(m=>m.loss??m.train_loss);
 if(losses.length>1){const min=Math.min(...losses),max=Math.max(...losses),points=losses.map((v,i)=>(i/(losses.length-1)*390+5)+","+(85-(v-min)/(max-min||1)*75)).join(" ");const el=document.createElementNS("http://www.w3.org/2000/svg","polyline");el.setAttribute("points",points);el.setAttribute("fill","none");el.setAttribute("stroke","#6f9b73");el.setAttribute("stroke-width","2");svg.append(el);}
 const last=stream.metrics.at(-1);if(last)$("metric-caption").textContent=[last.completed!=null?last.completed+" items":null,last.progress!=null?last.progress+" work units":null,Number.isFinite(last.loss)?"Loss "+last.loss.toFixed(4):null,Number.isFinite(last.tokens_per_second)?last.tokens_per_second.toFixed(1)+" tokens/s":null,last.cache_hit!=null?"Cache "+(last.cache_hit?"reused":"prepared"):null].filter(Boolean).join(" · ")||"Timing observation recorded";
 const gpu=[...stream.metrics].reverse().find(m=>m.gpus)?.gpus;
 if(Array.isArray(gpu))$("gpu-caption").textContent=gpu.filter(g=>g&&typeof g.uuid==="string"&&Number.isFinite(g.utilization)&&Number.isFinite(g.memory_used_mib)).map(g=>g.uuid.slice(0,12)+": "+g.utilization+"% · "+Math.round(g.memory_used_mib)+" MiB").join(" / ");
}
function fillForm(){
 const fields={"run-name":"name","run-kind":"kind","run-tasks":"task_count","run-gpus":"gpus","run-workers":"workers","run-concurrency":"concurrency","run-cpus":"cpus","run-completion":"completion"};
 for(const [id,key]of Object.entries(fields))$(id).value=draft[key]??"";
 $("run-command").value=JSON.stringify(draft.command,null,2);$("run-resume").checked=draft.resume_capable;$("run-auto").checked=draft.auto_continue;fillSetupFields(draft);
}
function fromForm(){return setupConfiguration({...draft,name:$("run-name").value,kind:$("run-kind").value,task_count:Number($("run-tasks").value),gpus:Number($("run-gpus").value),workers:Number($("run-workers").value),concurrency:Number($("run-concurrency").value),cpus:$("run-cpus").value?Number($("run-cpus").value):null,command:setupCommand(),completion:$("run-completion").value,resume_capable:$("run-resume").checked,auto_continue:$("run-auto").checked});}
async function setMode(next){
 try{
  if(next==="yaml"&&mode==="guided"){draft=fromForm();const result=await api("validate",{yaml:JSON.stringify(draft)});$("run-yaml").value=result.yaml;}
  if(next==="guided"&&mode==="yaml"){draft=(await api("validate",{yaml:$("run-yaml").value})).config;fillForm();}
  mode=next;$("guided-fields").hidden=mode!=="guided";$("yaml-fields").hidden=mode!=="yaml";$("guided-tab").classList.toggle("active",mode==="guided");$("yaml-tab").classList.toggle("active",mode==="yaml");$("run-error").textContent="";
 }catch(e){$("run-error").textContent=e.message;}
}
function loadTemplate(){
 const key=$("template-select").value;draft=structuredClone(boot.templates[key]);$("run-yaml").value=boot.template_yaml[key];fillForm();
 $("plan-preview").hidden=true;$("run-error").textContent="";
}
function openRun(){
 if(!profileId){openProfile(true);return;}
 draftId=crypto.randomUUID().replaceAll("-","");$("template-select").value=currentProfile().simulation?"simulator":"evaluation";mode="guided";loadTemplate();
 $("guided-fields").hidden=false;$("yaml-fields").hidden=true;$("guided-tab").classList.add("active");$("yaml-tab").classList.remove("active");$("run-dialog").showModal();
}
function configSource(){return mode==="yaml"?$("run-yaml").value:JSON.stringify(fromForm());}
async function saveRun(launch){
 const buttons=$("run-form").querySelectorAll("button");buttons.forEach(b=>b.disabled=true);
 let created;
 try{
  $('run-error').textContent='Saving configuration on the cluster…';draftId=draftId||crypto.randomUUID().replaceAll("-","");created=(await api("create",{profile:profileId,request_id:draftId,yaml:configSource()})).result;selectedId=created.id;
  if(launch){$("run-error").textContent="Refreshing queue estimates and submitting…";await api("launch",{profile:profileId,id:created.id});}
  $("run-dialog").close();await refresh(true);
 }catch(e){if(created){$("run-dialog").close();await refresh();notice(e.message+" The run was saved. Inspect its state before resuming.");}else{$("run-error").textContent=e.message;await refresh();}}
 finally{buttons.forEach(b=>b.disabled=false);}
}
const profileExample={name:"My cluster",ssh_host:"my-cluster",ssh_options:[],remote_root:"/absolute/path/to/workspace",python:"python3",account:"",qos:"",queue_threshold_seconds:10800,normal_min_seconds:7200,checkpoint_margin_seconds:180,poll_seconds:30,dev_permission:"",partitions:[{name:"gpu",role:"normal",time_limit:"02:00:00",concurrency_limit:1},{name:"dev_gpu",role:"development",time_limit:"00:30:00",concurrency_limit:1}]};
function openProfile(fresh=false){
 $("profile-id").value=fresh?"":profileId||"";$("profile-yaml").value=JSON.stringify(fresh||!currentProfile()?profileExample:currentProfile(),null,2);$("profile-error").textContent="";$("profile-dialog").showModal();
}
$("new-run").onclick=openRun;$("empty-new").onclick=openRun;
$("refresh").onclick=()=>refresh(true);
$("profile-select").onchange=()=>{profileId=$("profile-select").value;localStorage.setItem("cluster-profile",profileId);selectedId=null;runs=[];stream={key:"",offsets:{},buffer:{},metrics:[],log:""};$("detail").replaceChildren();renderRuns();renderProfile();refresh();};
$("profile-open").onclick=()=>openProfile();
$("new-profile").onclick=()=>{$("profile-id").value="";$("profile-yaml").value=JSON.stringify(profileExample,null,2);};
$("profile-form").onsubmit=async e=>{e.preventDefault();try{const id=$("profile-id").value;await api("profile",{id,yaml:$("profile-yaml").value});profileId=id;$("profile-dialog").close();await bootstrap();}catch(e){$("profile-error").textContent=e.message;}};
document.querySelectorAll("[data-close]").forEach(b=>b.onclick=()=>$(b.dataset.close).close());
$("template-select").onchange=loadTemplate;
$("guided-tab").onclick=()=>setMode("guided");$("yaml-tab").onclick=()=>setMode("yaml");
$("run-form").onsubmit=e=>{e.preventDefault();saveRun(true);};$("save-draft").onclick=()=>saveRun(false);
$("estimate").onclick=async()=>{const b=$("estimate");b.disabled=true;try{const p=(await api("discover",{profile:profileId,yaml:configSource()})).result;$("plan-preview").hidden=false;$("plan-preview").innerHTML='<h3>Queue comparison</h3>'+p.plans.map(x=>'<div class="plan-row">'+escapeHtml(x.name)+(p.recommendation?.name===x.name?" · recommended":"")+'<small>Queue '+fmt(x.queue_seconds)+' · Startup '+fmt(x.startup_seconds)+' · Finish '+fmt(x.finish_seconds)+(x.available?"":" · Unavailable")+'</small></div>').join("")+'<p class="form-help">Estimates are snapshots. Unknown waits stay unknown. Repeated dev allocations require a permission note in your profile.</p>';}catch(e){$("run-error").textContent=e.message;}finally{b.disabled=false;}};
bootstrap().catch(e=>notice(e.message));
setInterval(()=>{const ms=currentProfile()?.simulation?2000:Math.max(10000,(currentProfile()?.poll_seconds||30)*1000);if(!window.lastPoll||Date.now()-window.lastPoll>=ms){window.lastPoll=Date.now();refresh();}},2000);
