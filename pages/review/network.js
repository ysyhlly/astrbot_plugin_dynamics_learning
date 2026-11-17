// Presentation only. Whole-group metrics and stored relationship scores are unchanged.
// Inspired by the activity layout and label collision handling in Group Relation
// Analyzer (Hermes, MIT; see vendor/group_relations/NOTICE.md).
const svgNS = 'http://www.w3.org/2000/svg';
const palette = {positive:'#4c9983', negative:'#be5e76', mixed:'#bb914e', neutral:'#89939c', unknown:'#aaa0bd'};
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const num = v => Number.isFinite(Number(v)) ? Number(v) : 0;
function make(tag, attributes={}, text){
 const e=document.createElementNS(svgNS,tag);
 for(const [k,v] of Object.entries(attributes))e.setAttribute(k,String(v));
 if(text!==undefined)e.textContent=String(text);
 return e;
}
function tone(e){
 if(e.tone_state in palette)return e.tone_state;
 return num(e.affinity_confidence)<.2?'unknown':num(e.affinity)<-15?'negative':num(e.affinity)>15?'positive':'neutral';
}
function representative(edges, ids){
 const candidates=edges.filter(e=>ids.has(e.a)&&ids.has(e.b)&&e.a!==e.b&&
  (e.algorithm_version!==2||(num(e.recent_score)>=5&&num(e.directed_hits)+num(e.focused_hits)>0)))
  .slice().sort((a,b)=>num(b.score)-num(a.score)||String(a.a).localeCompare(String(b.a))||String(a.b).localeCompare(String(b.b)));
 const chosen=new Map(),degree=new Map(),parents=new Map([...ids].map(id=>[id,id]));
 const key=e=>JSON.stringify([e.a,e.b].sort());
 const add=e=>{if(chosen.has(key(e)))return;chosen.set(key(e),e);for(const id of [e.a,e.b])degree.set(id,(degree.get(id)||0)+1);};
 const root=id=>{while(parents.get(id)!==id)id=parents.get(id);return id;};
 // Keep bridges and well-supported negative ties, then representative partners.
 for(const e of candidates){const a=root(e.a),b=root(e.b);if(a!==b){parents.set(a,b);add(e);}}
 for(const e of candidates)if(num(e.affinity_confidence)>=.2&&num(e.negative_tendency??-num(e.affinity))>=20)add(e);
 for(const e of candidates)if((degree.get(e.a)||0)<3&&(degree.get(e.b)||0)<3)add(e);
 return [...chosen.values()];
}
function settle(nodes,width,height,passes=240){
 for(let pass=0;pass<passes;pass++){
  let largest=0;
  for(let i=0;i<nodes.length;i++)for(let j=0;j<i;j++){
   const a=nodes[i],b=nodes[j],ox=a.half+b.half+12-Math.abs(a.x-b.x);
   const oy=Math.min(a.y+a.bottom,b.y+b.bottom)-Math.max(a.y-a.r,b.y-b.r)+12;
   if(ox<=0||oy<=0)continue;
   largest=Math.max(largest,Math.min(ox,oy));
   if(ox<oy){const delta=ox*.51*(a.x>=b.x?1:-1);a.x+=delta;b.x-=delta;}
   else{const delta=oy*.51*(a.y+(a.bottom-a.r)/2>=b.y+(b.bottom-b.r)/2?1:-1);a.y+=delta;b.y-=delta;}
  }
  for(const n of nodes){n.x=clamp(n.x,n.half+18,width-n.half-18);n.y=clamp(n.y,n.r+18,height-n.bottom-18);}
  if(largest<.1)break;
 }
 // Resolve residual boundary jams with a bounded nearest-free search.
 const placed=[];
 const free=(n,x,y)=>placed.every(b=>Math.abs(x-b.x)>=n.half+b.half+10||
  Math.min(y+n.bottom,b.y+b.bottom)-Math.max(y-n.r,b.y-b.r)<=-10);
 for(const n of nodes){
  if(!free(n,n.x,n.y)){
   let best=null,distance=Infinity;
   for(let y=n.r+18;y<=height-n.bottom-18;y+=8)for(let x=n.half+18;x<=width-n.half-18;x+=8){
    const d=(x-n.x)**2+(y-n.y)**2;if(d<distance&&free(n,x,y)){best={x,y};distance=d;}
   }
   if(best){n.x=best.x;n.y=best.y;}
   else{
    const cellW=2*Math.max(...nodes.map(v=>v.half))+16,cellH=Math.max(...nodes.map(v=>v.r+v.bottom))+16;
    const cols=Math.max(1,Math.floor((width-36)/cellW)),cells=[];
    for(let i=0;i<nodes.length;i++)cells.push({x:(width-cols*cellW)/2+(i%cols+.5)*cellW,y:18+Math.floor(i/cols)*cellH+cellH/2-10});
    cells.sort((a,b)=>Math.hypot(a.x-width/2,a.y-height/2)-Math.hypot(b.x-width/2,b.y-height/2));
    nodes.forEach((v,i)=>Object.assign(v,cells[i]));return;
   }
  }
  placed.push(n);
 }
}
function layout(nodes,edges,width,height){
 let seed=2166136261;
 for(const ch of nodes.map(n=>n.id).join('|'))seed=Math.imul(seed^ch.charCodeAt(0),16777619)>>>0;
 const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
 const placed=new Map(),outerX=width/2-65,outerY=height/2-70;
 for(let i=0;i<nodes.length;i++){
  const n=nodes[i],target=.85*Math.sqrt(i/Math.max(1,nodes.length-1));
  const neighbours=edges.filter(e=>e.a===n.id||e.b===n.id).map(e=>placed.get(e.a===n.id?e.b:e.a)).filter(Boolean);
  let best=-Infinity;
  for(let attempt=0;attempt<100;attempt++){
   const angle=random()*Math.PI*2,radius=clamp(target+(random()-.5)*.5,0,1);
   const x=clamp(width/2+Math.cos(angle)*outerX*radius,n.half+18,width-n.half-18);
   const y=clamp(height/2+Math.sin(angle)*outerY*radius,n.r+18,height-n.bottom-18);
   let score=nodes.length===1?0:1000;
   for(const other of placed.values())score=Math.min(score,Math.hypot((x-other.x)*.8,y-other.y)-n.r-other.r-24);
   score-=80*Math.abs(Math.hypot((x-width/2)/outerX,(y-height/2)/outerY)-target);
   if(neighbours.length)score-=.15*neighbours.reduce((s,m)=>s+Math.hypot(m.x-x,m.y-y),0)/neighbours.length;
   if(score>best){best=score;n.x=x;n.y=y;}
  }
  placed.set(n.id,n);
 }
 settle(nodes,width,height);
}

export class RelationNetwork{
 constructor(svg,selectMember,remote=null){
  this.remote=remote;
  this.svg=svg;this.selectMember=selectMember;this.graph=null;this.selected=null;this.nodes=[];this.edges=[];this.width=0;
  this.limit=document.querySelector('#network-limit');this.search=document.querySelector('#network-search');
  this.limit.value=window.innerWidth<650?'20':'40';
  this.limit.onchange=()=>{if(!this.remote||!this.remote('',Number(this.limit.value)))this.render();};
  document.querySelector('#network-find').onclick=()=>this.find();
  this.search.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();this.find();}};
  document.querySelector('#network-reset').onclick=()=>{this.selected=null;this.search.value='';this.selectMember(null);if(!this.remote||!this.remote('',Number(this.limit.value)))this.render();};
  this.observer=new ResizeObserver(()=>{if(this.graph&&svg.getBoundingClientRect().width>0&&Math.abs(svg.getBoundingClientRect().width-this.width)>2)this.render();});
  this.observer.observe(svg);
 }
 setData(graph){
  this.graph=graph;this.selected=graph.focus||null;this.search.value=this.selected||'';this.selectMember(null);
  const list=document.querySelector('#network-members');list.replaceChildren();
  for(const [id,user] of Object.entries(graph.users||{})){const o=document.createElement('option');o.value=id;o.label=user.name||id;list.append(o);}
  this.render();
  if(this.selected)this.selectMember(this.selected);
 }
 find(){
  if(!this.graph)return;
  const query=this.search.value.trim().toLowerCase();if(!query)return;
  if(this.remote&&this.remote(query,Number(this.limit.value)))return;
  const entries=Object.entries(this.graph.users||{}),found=entries.find(([id])=>id.toLowerCase()===query)||entries.find(([,u])=>String(u.name||'').toLowerCase()===query)||entries.find(([id,u])=>`${id} ${u.name||''}`.toLowerCase().includes(query));
  if(!found){document.querySelector('#network-summary').textContent='没有找到该成员，请输入完整账号或昵称。';return;}
  this.selected=found[0];this.render();this.selectMember(this.selected);
 }
 render(){
  if(!this.graph)return;
  const svg=this.svg,g=this.graph,width=Math.max(300,svg.getBoundingClientRect().width);this.width=width;
  const all=Object.entries(g.users||{}).sort((a,b)=>num(b[1].count)-num(a[1].count)||a[0].localeCompare(b[0]));
  let visible=all.slice(0,Number(this.limit.value));
  if(this.selected){
   const ids=new Set([this.selected]);
   for(const e of (g.all_relations||[]).filter(e=>e.a===this.selected||e.b===this.selected).sort((a,b)=>num(b.score)-num(a.score))){ids.add(e.a);ids.add(e.b);if(ids.size>=Number(this.limit.value))break;}
   visible=[...all.filter(([id])=>ids.has(id)),...all.filter(([id])=>!ids.has(id))].slice(0,Number(this.limit.value));
  }
  const measure=document.createElement('canvas').getContext('2d');measure.font='12px '+getComputedStyle(svg).fontFamily;
  this.nodes=visible.map(([id,u])=>{
   const name=String(u.name||id),chars=Array.from(name),limit=width<500?4:6,label=chars.length>limit?chars.slice(0,limit).join('')+'…':name;
   const r=11+Math.min(10,Math.log1p(Math.max(0,num(u.count)))*1.5);
   return {id,name,label,r,half:Math.max(r,measure.measureText(label).width/2),bottom:r+23,count:num(u.count)};
  });
  const cellW=2*Math.max(35,...this.nodes.map(n=>n.half))+16,cellH=Math.max(70,...this.nodes.map(n=>n.r+n.bottom))+16;
  const height=Math.max(width<500?560:620,Math.ceil(visible.length/Math.max(1,Math.floor((width-36)/cellW)))*cellH+50);
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.style.height=`${height}px`;svg.replaceChildren();this.height=height;
  const ids=new Set(this.nodes.map(n=>n.id));
  this.edges=representative(g.all_relations||[],ids);
  // In member focus show every displayed direct partner, even if the overview pruned it.
  if(this.selected){const chosen=new Set(this.edges.map(e=>JSON.stringify([e.a,e.b].sort())));for(const e of g.all_relations||[]){const k=JSON.stringify([e.a,e.b].sort());if((e.a===this.selected||e.b===this.selected)&&ids.has(e.a)&&ids.has(e.b)&&!chosen.has(k)){this.edges.push(e);chosen.add(k);}}}
  layout(this.nodes,this.edges,width,height);
  this.lines=this.edges.map(e=>{const line=make('line',{'data-a':e.a,'data-b':e.b,stroke:palette[tone(e)],'stroke-width':.8+2.5*Math.sqrt(clamp(num(e.score)/100,0,1)),'stroke-dasharray':tone(e)==='unknown'?'4 6':'none'});svg.append(line);return {e,line};});
  this.groups=this.nodes.map(n=>{
   const group=make('g',{'data-member':n.id,role:'button',tabindex:0,'aria-label':n.name,'aria-pressed':n.id===this.selected});
   const circle=make('circle',{r:n.r,fill:'#3d7b65',stroke:'var(--network-surface)','stroke-width':2});
   const label=make('text',{y:n.r+17,'text-anchor':'middle','font-size':12,class:'network-label'},n.label);
   group.append(circle,label,make('title',{},`${n.name} · ${n.id} · ${n.count} 条近期消息`));svg.append(group);
   const select=()=>{this.selected=n.id;this.selectMember(n.id);this.paint();};
   let drag=null;
   group.onpointerdown=e=>{if(e.button!==0)return;e.preventDefault();drag={x:e.clientX,y:e.clientY,nx:n.x,ny:n.y,moved:false};group.setPointerCapture(e.pointerId);};
   group.onpointermove=e=>{if(!drag)return;const rect=svg.getBoundingClientRect(),dx=(e.clientX-drag.x)*width/rect.width,dy=(e.clientY-drag.y)*height/rect.height;drag.moved ||= Math.hypot(dx,dy)>4;n.x=clamp(drag.nx+dx,n.half+18,width-n.half-18);n.y=clamp(drag.ny+dy,n.r+18,height-n.bottom-18);this.paint();};
   group.onpointerup=e=>{if(!drag)return;const moved=drag.moved;drag=null;group.releasePointerCapture(e.pointerId);if(moved){settle(this.nodes,width,height,120);this.paint();}else select();};
   group.onpointercancel=()=>{drag=null;};
   group.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};
   return {n,group};
  });
  if(!this.nodes.length)svg.append(make('text',{x:width/2,y:height/2,'text-anchor':'middle',fill:'currentColor'},'等待群聊互动数据'));
  document.querySelector('#network-summary').textContent=`显示 ${this.nodes.length} / ${g.total_members??all.length} 位成员 · ${this.edges.length} 条代表性连线${this.selected?' · 当前聚焦成员':''}。全群统计不受显示数量影响。`;
  this.paint();
 }
 paint(){
  const positions=new Map(this.nodes.map(n=>[n.id,n])),neighbours=new Set([this.selected]);
  for(const {e,line} of this.lines||[]){
   const a=positions.get(e.a),b=positions.get(e.b),related=e.a===this.selected||e.b===this.selected;
   if(related){neighbours.add(e.a);neighbours.add(e.b);}
   for(const [key,value] of Object.entries({x1:a.x,y1:a.y,x2:b.x,y2:b.y,opacity:this.selected?(related?.85:.07):.35}))line.setAttribute(key,value);
  }
  for(const {n,group} of this.groups||[]){group.setAttribute('transform',`translate(${n.x},${n.y})`);group.setAttribute('opacity',!this.selected||neighbours.has(n.id)?1:.25);group.setAttribute('aria-pressed',n.id===this.selected);group.querySelector('circle').setAttribute('fill',n.id===this.selected?'#b67e39':'#3d7b65');}
 }
}
