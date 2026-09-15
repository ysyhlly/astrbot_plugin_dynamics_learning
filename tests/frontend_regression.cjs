const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

// Load the shipped script order. The bridge is provided by the host below;
// only automatic startup is omitted so each scenario controls its requests.
function fixture() {
  const elements = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, {
      id, innerHTML: '', textContent: '', disabled: false, dataset: {}, hidden: false,
      classList: {toggle(){}, add(){}, remove(){}},
      setAttribute(){}, removeAttribute(){}, appendChild(){},
    });
    return elements.get(id);
  };
  const sandbox = {console, setTimeout, clearTimeout, URLSearchParams,
    window: {location: {hash: ''}},
    document: {getElementById: get, querySelectorAll: () => [],
      createElement: () => ({addEventListener(){}, setAttribute(){}})}};
  vm.createContext(sandbox);
  const directory = path.join(__dirname, '../pages/learning');
  const html = fs.readFileSync(path.join(directory, 'index.html'), 'utf8');
  const scripts = [...html.matchAll(/<script\b[^>]*\bsrc="([^"]+)"[^>]*>/g)]
    .map(match => match[1]).filter(src => !/bridge|^https?:/i.test(src));
  assert.ok(scripts.length > 1, 'the regression harness must exercise the modular page');
  for (const script of scripts) {
    const source = fs.readFileSync(path.join(directory, script), 'utf8');
    vm.runInContext(source.replace(/\bmain\(\);\s*$/, ''), sandbox, {filename: script});
  }
  return {get, run: code => vm.runInContext(code, sandbox),
    api: apiGet => {sandbox.window.AstrBotPluginPage = {apiGet};}};
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => {resolve = yes; reject = no;});
  return {promise, resolve, reject};
}
const tick = () => new Promise(resolve => setImmediate(resolve));
const A = 'a'.repeat(64), B = 'b'.repeat(64);

(async () => {
  const f = fixture();
  for (const fn of ['num', 'pct', 'humanDuration', 'wallMoment', 'windowDeadline']) {
    assert.equal(f.run(`${fn}(null)`), '—');
    assert.equal(f.run(`${fn}(undefined)`), '—');
  }
  assert.equal(f.run('pct(0)'), '0.0%');
  f.api(() => new Promise(() => {}));
  await assert.rejects(f.run('call("slow", {timeoutMs: 5})'), /超时/);
  f.api(() => Promise.reject(new Error('offline')));
  await assert.rejects(f.run('loadReplyReview()'), /offline/);
  await assert.rejects(f.run('loadAnnotationWindow()'), /offline/);
  let params;
  f.api((_, p) => {params = p; return {review: {rows: []}};});
  await f.run('loadReplyReview()');
  assert.equal(params.refresh, undefined, 'opening review must allow cached results');
  await f.run('loadReplyReview({refresh:true})');
  assert.equal(params.refresh, 1, 'explicit regeneration must reach the API');
  f.api(() => ({state: 'unavailable', reason: 'no model'}));
  await assert.rejects(f.run('loadReview()'), /no model/);
  f.run('renderPolicies({rows:[{version:"v1",available_actions:["ignore"]}]})');
  assert.match(f.get('policies').innerHTML, /data-action="ignore"/);
  assert.doesNotMatch(f.get('policies').innerHTML, /data-action="accept"/);

  // Resolve the second session first, then deliver the stale first response.
  const profiles = fixture(), pendingProfiles = [];
  profiles.api((endpoint, p) => {
    const request = deferred(); pendingProfiles.push({endpoint, params: p, ...request});
    return request.promise;
  });
  const firstProfile = profiles.run(`loadScope('${A}')`);
  const secondProfile = profiles.run(`loadScope('${B}')`);
  await tick();
  assert.equal(pendingProfiles[0].params.id, A);
  assert.equal(pendingProfiles[1].params.id, B);
  pendingProfiles[1].resolve({profile: {labelled_samples: 22}, note: 'new session evidence'});
  await secondProfile;
  pendingProfiles[0].resolve({profile: {labelled_samples: 11}, note: 'stale session evidence'});
  await firstProfile;
  assert.equal(profiles.run('state.selectedScope'), B);
  assert.match(profiles.get('scopeDetail').innerHTML, /new session evidence/);
  assert.doesNotMatch(profiles.get('scopeDetail').innerHTML, /stale session evidence/);
  for (const view of ['data', 'evaluation', 'shadow', 'policies']) {
    profiles.run(`setView('${view}')`);
    assert.equal(profiles.get('scopeBadge').textContent, '全局数据');
  }
  profiles.run('setView("workspace")');
  assert.match(profiles.get('scopeBadge').textContent, /会话/);

  const pages = [];
  profiles.api((endpoint, p) => {
    const request = deferred(); pages.push({endpoint, params: p, ...request});
    return request.promise;
  });
  const page2 = profiles.run('loadSamples(2,"topic")');
  const page3 = profiles.run('loadSamples(3,"topic")');
  await tick();
  assert.equal(pages[0].params.scope, B, 'samples must receive the entire selected hash');
  assert.equal(pages[1].params.page, 3);
  pages[1].resolve({rows: [{msg_id: 'latest-page'}], page: 3, total: 101});
  await page3;
  pages[0].resolve({rows: [{msg_id: 'stale-page'}], page: 2, total: 101});
  await page2;
  assert.match(profiles.get('samples').innerHTML, /latest-page/);
  assert.doesNotMatch(profiles.get('samples').innerHTML, /stale-page/);
  assert.equal(profiles.get('btnSamplesPrev').disabled, false);
  assert.equal(profiles.get('btnSamplesNext').disabled, true);
  profiles.api((_, p) => ({rows: [], page: p.page, total: 101}));
  await profiles.run('loadSamples(2,"topic")');
  assert.equal(profiles.get('btnSamplesNext').disabled, false);

  await profiles.run('withBusy($("btnSamplesNext"), "加载中…", () => loadSamples(3,"topic"))');
  assert.equal(profiles.get('btnSamplesNext').disabled, true,
    'button busy cleanup must preserve the final-page navigation boundary');

  // A pending sample page must not leak into another session after selection.
  const staleSamples = deferred();
  profiles.api(endpoint => endpoint === 'samples' ? staleSamples.promise : {profile: {}});
  const previousSessionPage = profiles.run('loadSamples(1,"topic")');
  await tick();
  await profiles.run(`loadScope('${A}')`);
  staleSamples.resolve({rows: [{msg_id: 'wrong-session-evidence'}], page: 1, total: 1});
  await previousSessionPage;
  assert.doesNotMatch(profiles.get('samples').innerHTML, /wrong-session-evidence/);
  assert.equal(profiles.run('state.samples'), null);

  // Ordinary refresh is read-only, coalesced, and preserves successful content
  // if a later request fails. Real renderers are used throughout this test.
  const refresh = fixture(), calls = [];
  refresh.api(endpoint => {
    calls.push(endpoint);
    return endpoint === 'overview' ? {dataset: {samples: 7}} : {rows: []};
  });
  await refresh.run('refresh()');
  const previous = refresh.get('quality').innerHTML;
  calls.length = 0;
  const requests = new Map();
  refresh.api(endpoint => {
    calls.push(endpoint); const request = deferred(); requests.set(endpoint, request);
    return request.promise;
  });
  const first = refresh.run('refresh()'), second = refresh.run('refresh()');
  assert.equal(first, second, 'overlapping refreshes should share completion');
  const failedRefresh = assert.rejects(first, /未能更新/);
  await tick();
  assert.deepEqual(calls.slice().sort(), ['overview', 'quality', 'scopes']);
  requests.get('overview').resolve({dataset: {samples: 9}});
  requests.get('scopes').resolve({rows: []});
  requests.get('quality').reject(new Error('quality failed'));
  await failedRefresh;
  assert.match(refresh.get('quality').innerHTML, /quality failed/);
  assert.ok(refresh.get('quality').innerHTML.includes(previous), 'failed refresh retains the previous report');
  assert.equal(refresh.run('state.overview.dataset.samples'), 9, 'one failure must not discard other successful updates');
  assert.ok(!calls.some(endpoint => ['review', 'reply_review', 'analyze', 'ingest'].includes(endpoint)));
  // Import/analysis invalidates requests already in flight. An ensuing read
  // must fetch a new snapshot and never paint the pre-mutation response.
  const invalidated = fixture(), snapshots = [];
  invalidated.api(() => {
    const request = deferred(); snapshots.push(request); return request.promise;
  });
  const oldRead = invalidated.run('loadResource("overview")');
  await tick();
  invalidated.run('resources.overview.stale = true; resources.overview.epoch = (resources.overview.epoch || 0) + 1');
  const newRead = invalidated.run('loadResource("overview")');
  snapshots[0].resolve({dataset: {samples: 9}});
  await oldRead;
  await tick();
  assert.equal(invalidated.run('state.overview'), null,
    'the outdated response must never enter displayed state');
  assert.equal(invalidated.get('overview').innerHTML, '');
  assert.equal(snapshots.length, 2, 'a read following invalidation must request fresh data');
  snapshots[1].resolve({dataset: {samples: 11}});
  await newRead;
  assert.equal(invalidated.run('state.overview.dataset.samples'), 11);
  assert.match(invalidated.get('overview').innerHTML, />11</);

  f.run('renderEvaluation({evaluation:{verdict:"accepted"}, promotion:{verdict:"rejected", reasons:["cross-validation failed"]}})');
  const heading = f.get('eval').innerHTML.match(/<h3>([\s\S]*?)<\/h3>/)[1];
  assert.match(heading, /综合采纳结论/);
  assert.match(heading, /拒绝/);
  assert.doesNotMatch(heading, /可采纳/);
  assert.match(f.get('eval').innerHTML, /cross-validation failed/);
  console.log('frontend regression checks passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
