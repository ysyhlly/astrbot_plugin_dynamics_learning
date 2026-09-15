const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const elements = new Map();
const get = id => {
  if (!elements.has(id)) elements.set(id, {innerHTML: '', textContent: '', disabled:false, classList:{toggle(){},add(){},remove(){}}, setAttribute(){},appendChild(){}});
  return elements.get(id);
};
const sandbox = {console, setTimeout, clearTimeout, window:{}, document:{getElementById:get, querySelectorAll:()=>[], createElement:()=>({addEventListener(){},setAttribute(){}})}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../pages/learning/app.js'), 'utf8').replace(/main\(\);\s*$/, ''), sandbox);
const run = code => vm.runInContext(code,sandbox);
(async () => {
  for (const fn of ['num','pct','humanDuration','wallMoment','windowDeadline']) {
    assert.equal(run(`${fn}(null)`),'—');
    assert.equal(run(`${fn}(undefined)`),'—');
  }
  assert.equal(run('pct(0)'), '0.0%');
  sandbox.window.AstrBotPluginPage={apiGet:()=>new Promise(()=>{})};
  await assert.rejects(run('call("slow", {timeoutMs: 5})'), /超时/);
  sandbox.window.AstrBotPluginPage={apiGet:()=>Promise.reject(new Error('offline'))};
  await assert.rejects(run('loadReplyReview()'), /offline/);
  await assert.rejects(run('loadAnnotationWindow()'), /offline/);
  let params;
  sandbox.window.AstrBotPluginPage={apiGet:(_,p)=>{params=p; return {review:{rows:[]}};}};
  await run('loadReplyReview()'); assert.equal(params.refresh,undefined);
  sandbox.window.AstrBotPluginPage={apiGet:()=>({state:'unavailable',reason:'no model'})};
  await assert.rejects(run('loadReview()'),/no model/);
  sandbox.window.AstrBotPluginPage={apiGet:(_,p)=>({rows:[],page:p.page,total:101})};
  await run('loadSamples(2,"topic")');
  assert.equal(get('btnSamplesPrev').disabled,false); assert.equal(get('btnSamplesNext').disabled,false);
  await run('loadSamples(3,"topic")'); assert.equal(get('btnSamplesNext').disabled,true);
  run('renderPolicies({rows:[{version:"v1",available_actions:["ignore"]}]})');
  assert.match(get('policies').innerHTML,/data-action="ignore"/); assert.doesNotMatch(get('policies').innerHTML,/data-action="accept"/);
  run('for (const name of ["renderOverview","renderWindow","renderErrors","renderRecommendations","renderCandidates","renderTuning","renderEvaluation","renderPolicies","renderQuality","renderAttribution","renderShadow"]) globalThis[name] = () => {}; loadReview = async () => {}; loadAnnotationWindow = async () => {};');
  let calls=0;
  sandbox.window.AstrBotPluginPage={apiGet:endpoint=>{calls++; return endpoint==='quality'?Promise.reject(new Error('quality failed')):{report:{}};}};
  const first=run('refresh()'), second=run('refresh()'); assert.equal(first,second);
  await assert.rejects(first,/quality failed/); assert.equal(calls,6);
  assert.match(get('quality').innerHTML,/quality failed/);
  console.log('frontend regression checks passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
