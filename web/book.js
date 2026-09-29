/* Shared by the comparison and per-run dashboards. No report or board image payloads. */
(() => {
const css = document.createElement('style');
css.textContent = `
.book-tabs,.book-controls,.book-pages{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 14px}
.book-tabs button,.book-panel button,.book-panel input,.book-panel select{font:inherit;font-size:12px;color:inherit;background:var(--raised,#f3f5f7);border:1px solid var(--axis,#c5d0d7);border-radius:4px;padding:4px 7px}
.book-tabs button,.book-panel button{cursor:pointer}.book-tabs [aria-pressed=true]{border-color:#3987e5;color:var(--ink,#175e83)}
.book-panel{background:var(--surface,#fff);border:1px solid var(--ring,#d8e0e5);border-radius:6px;padding:14px;margin-bottom:24px}
.book-panel[hidden],.book-overview[hidden],[hidden].book-tabs,.book-controls [hidden]{display:none}.book-panel label{font-size:12px;display:flex;align-items:center;gap:5px}
.book-panel input[type=number]{width:68px}.book-panel button:disabled{opacity:.4;cursor:default}
.book-scroll{overflow:auto;max-height:70vh}.book-panel table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
.book-panel th,.book-panel td{padding:3px 10px;border-bottom:1px solid var(--grid,#e6ebef);white-space:nowrap;text-align:right}
.book-panel th{position:sticky;top:0;background:var(--surface,#fff);z-index:1}.book-panel th button{border:0;background:none;padding:6px 0}.book-panel th:first-child,.book-panel td:first-child{text-align:left;width:64px}
.book-panel td svg{display:block}.book-pages{margin:12px 0 0;font-size:12px;font-variant-numeric:tabular-nums}.book-error{color:#d95926;font-size:12px}
.book-mini .move-order{visibility:hidden;pointer-events:none}.book-mini:hover .move-order,.book-mini:focus .move-order{visibility:visible}
.book-dag{overflow:auto;max-height:65vh}.book-dag svg{display:block}.book-node{cursor:pointer}.book-node:focus{outline:none}.book-node:focus .node-border{stroke-width:3}
`;
document.head.append(css);
const element = (tag, text, attrs={}) => {const e=document.createElement(tag);if(text!=null)e.textContent=text;for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e};
const svg = (tag, attrs={}) => {const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e};
const statusColour = status => status==='opening'?'#199e70':status==='retired'?'#c98500':'#898781';
const number = v => v==null?'·':Number.isInteger(v)?String(v):v.toFixed(2);
const percent = v => v==null?'·':(100*v).toFixed(1);
const timestamp = v => v==null?'·':new Date(v*1000).toISOString().slice(0,16).replace('T',' ');
const columns = [
  ['games','Games','Book games through this canonical prefix',number],
  ['p1_win_rate','P1 %','First-player wins / decisive games; capped games excluded',percent],
  ['skew_z','z','Signed colour skew: (P1 wins − P2 wins) / √decisive games. Descriptive binomial score, not a paired significance test',number],
  ['decisive_share','Dec. %','Decisive games / all book games',percent],
  ['median_plies','Median','Median total placements in matching report games, including capped games',number],
  ['mean_plies','Mean','Mean total placements in matching report games, including capped games',number],
  ['depth','Depth','Opening placements, including the first origin stone',number],
  ['champion_probability','P %','Probability of reaching this canonical position under the champion',v=>v==null?'·':(100*v).toPrecision(3)],
  ['status','S','Green: opening; amber: retired; grey: prefix',v=>'●'],
  ['created_at','Created','Book creation or adoption time, UTC',timestamp],
  ['retired_at','Retired','Retirement time, UTC',timestamp],
  ['reason','Reason','Retirement reason',v=>v||'·']
];

function miniature(moves) {
  // Same pointy hex projection and paired turn ownership as web/index.html and hexcrop.
  const project=([q,r])=>[Math.sqrt(3)*(q+r/2),1.5*r], cells=new Map();
  for(const [q,r] of moves.length?moves:[[0,0]])for(const [dq,dr] of [[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,-1],[-1,1]])cells.set(`${q+dq},${r+dr}`,[q+dq,r+dr]);
  const points=[...cells.values()].map(project),xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  const loX=Math.min(...xs)-1,loY=Math.min(...ys)-1,w=Math.max(...xs)-loX+1,h=Math.max(...ys)-loY+1;
  const out=svg('svg',{width:64,height:64,viewBox:`${loX} ${loY} ${w} ${h}`,class:'book-mini',role:'img','aria-label':`${moves.length} opening placements`,tabindex:0});
  for(const [x,y] of points){const corners=Array.from({length:6},(_,i)=>{const a=(60*i-30)*Math.PI/180;return `${x+.96*Math.cos(a)},${y+.96*Math.sin(a)}`}).join(' ');out.append(svg('polygon',{points:corners,fill:'none',stroke:'#898781','stroke-opacity':.45,'stroke-width':.045}))}
  moves.forEach((move,i)=>{const [x,y]=project(move),p=Math.floor((i+1)/2)%2;out.append(svg('circle',{cx:x,cy:y,r:.72,fill:p?'#d95926':'#3987e5'}));const text=svg('text',{x,y:y+.23,'text-anchor':'middle',fill:'#fff','font-size':.7,class:'move-order'});text.textContent=i+1;out.append(text)});
  return out;
}

class OpeningBook {
  constructor(overview, single=false) {
    this.overview=overview;overview.classList.add('book-overview');this.single=single;this.runs=[];this.dags=new Map();this.expanded=new Map();this.request=0;
    this.tabs=element('nav',null,{class:'book-tabs','aria-label':'Dashboard views'});
    this.panel=element('div',null,{class:'book-panel',hidden:''});
    overview.before(this.tabs);overview.after(this.panel);
    for(const [view,label] of [['charts','Charts'],['book','Book'],['retired','Retired'],['dag','DAG']]){const b=element('button',label);b.dataset.view=view;b.onclick=()=>this.change({view,page:1});this.tabs.append(b)}
    this.controls=element('div',null,{class:'book-controls'});this.panel.append(this.controls);
    this.runSelect=element('select',null,{'aria-label':'Book run'});this.runSelect.onchange=()=>this.change({run:this.runSelect.value,page:1});this.controls.append(this.runSelect);
    this.filters=[];
    this.status=this.select('Status','status',[['','All'],['opening','Opening'],['retired','Retired'],['prefix','Prefix']]);
    this.reason=this.select('Reason','reason',[['','All'],['probability','Probability'],['skew','Skew'],['replaced','Replaced']]);
    this.minGames=this.input('Games ≥','min_games',0,0);this.depth=this.input('Depth','depth','',0);
    this.colour=this.input('Colour decides','colour_decides',false,0,'checkbox');
    this.minDecisive=this.input('Decisive ≥','min_decisive',10,1);
    this.pageSize=this.select('Rows','page_size',[['25','25'],['50','50'],['100','100'],['200','200']]);
    this.error=element('div',null,{class:'book-error',role:'status'});this.content=element('div');this.pages=element('div',null,{class:'book-pages'});this.panel.append(this.error,this.content,this.pages);
    window.addEventListener('hashchange',()=>this.render());
    this.render();
  }
  select(label,key,options){const e=element('select',null,{'aria-label':label});for(const [v,t] of options)e.append(element('option',t,{value:v}));this.control(label,key,e);return e}
  input(label,key,value,min,type='number'){const e=element('input',null,{type,min,step:1,'aria-label':label});e.value=value;this.control(label,key,e);return e}
  control(label,key,e){const wrap=element('label',label);wrap.append(e);this.controls.append(wrap);this.filters.push([key,e,wrap]);e.onchange=()=>{if(!e.checkValidity()){e.reportValidity();return}this.change({[key]:e.type==='checkbox'?(e.checked?'1':'0'):e.value,page:1})}}
  state(){const q=new URLSearchParams(location.hash.slice(1)),get=(k,d)=>q.get('book_'+k)??d;return {view:get('view','charts'),run:get('run',this.runs[0]||''),sort:get('sort','games'),direction:get('direction','desc'),page:get('page','1'),page_size:get('page_size','50'),status:get('status',''),reason:get('reason',''),min_games:get('min_games','0'),depth:get('depth',''),colour_decides:get('colour_decides','0'),min_decisive:get('min_decisive','10')}}
  change(values){const q=new URLSearchParams(location.hash.slice(1));for(const [k,v] of Object.entries({...this.state(),...values}))q.set('book_'+k,v);const hash=q.toString();if(location.hash.slice(1)===hash)this.render();else location.hash=hash}
  setRuns(names){
    const changed=names.join('\n')!==this.runs.join('\n');
    if(changed){this.runs=names;this.runSelect.replaceChildren(...names.map(n=>element('option',n,{value:n})))}
    if(names.length&&!names.includes(this.state().run)){this.change({run:names[0],page:1});return}
    if(changed)this.render();else if(['book','retired'].includes(this.state().view))this.render(true);
  }
  query(state){const q=new URLSearchParams(state);q.delete('view');if(this.single){q.delete('run');const run=new URLSearchParams(location.search).get('run');if(run)q.set('run',run)}return q}
  async fetch(path,q,signal){const res=await fetch(path+'?'+q,{signal});if(!res.ok)throw Error(`Book: HTTP ${res.status}`);return res.json()}
  async render(refresh=false){
    if(refresh&&this.loading)return;
    const s=this.state(),id=++this.request;this.abort?.abort();this.abort=new AbortController();const signal=this.abort.signal;
    const isChart=s.view==='charts',isDag=s.view==='dag';this.overview.hidden=!isChart;this.panel.hidden=isChart;
    for(const b of this.tabs.children)b.setAttribute('aria-pressed',b.dataset.view===s.view);
    if(isChart){window.dispatchEvent(new Event('resize'));return}
    if(!refresh){
      this.runSelect.hidden=this.single;this.runSelect.value=s.run;
      for(const [key,e,wrap] of this.filters){if(e.type==='checkbox')e.checked=s[key]==='1';else e.value=s[key];wrap.hidden=isDag||(s.view==='retired'&&key==='status')}
      this.content.replaceChildren();this.pages.replaceChildren();this.error.textContent='';this.pageSignature=null;
    }
    if(!this.runs.length)return;
    this.loading=true;
    try{
      if(isDag){
        const q=this.query({run:s.run}),key=q.toString();let data=this.dags.get(key);
        if(!data){data=await this.fetch('/api/book/dag',q,signal);this.dags.set(key,data)}
        if(id===this.request)this.drawDag(data,s);return;
      }
      if(s.view==='retired')s.status='retired';
      const data=await this.fetch('/api/book',this.query(s),signal);if(id!==this.request)return;
      this.error.textContent='';const signature=JSON.stringify(data);
      if(refresh&&signature===this.pageSignature)return;
      this.pageSignature=signature;
      const old=this.content.querySelector('.book-scroll'),scroll=[old?.scrollLeft||0,old?.scrollTop||0],table=this.table(data.rows,s);
      this.content.replaceChildren(table);table.scrollLeft=scroll[0];table.scrollTop=scroll[1];this.pages.replaceChildren();
      const count=Math.max(1,Math.ceil(data.total/data.page_size));
      if(data.page>count){this.change({page:count});return}
      for(const [label,page,disabled] of [['‹',data.page-1,data.page===1],['›',data.page+1,data.page>=count]]){const b=element('button',label,{'aria-label':label==='‹'?'Previous page':'Next page'});b.disabled=disabled;b.onclick=()=>this.change({page});this.pages.append(b)}
      this.pages.append(element('span',`${data.page} / ${count} · ${data.total}`,{title:'Page / pages · matching positions'}));
    }catch(e){if(e.name!=='AbortError'&&id===this.request)this.error.textContent=e.message}
    finally{if(id===this.request)this.loading=false}
  }
  table(rows,s){
    const wrap=element('div',null,{class:'book-scroll'}),table=element('table'),head=element('thead'),tr=element('tr');tr.append(element('th',''));
    for(const [key,label,title] of columns){const th=element('th',null,{'aria-sort':s.sort===key?(s.direction==='asc'?'ascending':'descending'):'none'}),b=element('button',label+(s.sort===key?(s.direction==='asc'?' ↑':' ↓'):''),{title});b.onclick=()=>this.change({sort:key,direction:s.sort===key&&s.direction==='desc'?'asc':'desc',page:1});th.append(b);tr.append(th)}
    head.append(tr);table.append(head);const body=element('tbody');
    for(const row of rows){const tr=element('tr'),board=element('td');board.title=row.key;board.append(miniature(row.moves));tr.append(board);
      for(const [key,,title,format] of columns){const td=element('td',format(row[key]),{title});if(key==='status'){td.style.color=statusColour(row.status);td.title=row.status||'prefix'}if(key==='median_plies'||key==='mean_plies')td.title=title+` · ${row.report_games} report games`;tr.append(td)}body.append(tr)}
    table.append(body);wrap.append(table);return wrap;
  }
  drawDag(data,s){
    this.content.replaceChildren();
    const nodes=new Map(data.nodes.map(n=>[n.key,n])),children=new Map();
    for(const n of data.nodes)for(const p of n.parents)if(nodes.has(p)){if(!children.has(p))children.set(p,[]);children.get(p).push(n.key)}
    const roots=data.nodes.filter(n=>!n.parents.some(p=>nodes.has(p)));
    if(!this.expanded.has(s.run))this.expanded.set(s.run,new Set(roots.map(n=>n.key)));
    const expanded=this.expanded.get(s.run),visible=new Set(),visit=key=>{if(visible.has(key))return;visible.add(key);if(expanded.has(key))for(const child of children.get(key)||[])visit(child)};roots.forEach(n=>visit(n.key));
    const layers=new Map();for(const key of visible){const n=nodes.get(key);if(!layers.has(n.depth))layers.set(n.depth,[]);layers.get(n.depth).push(n)}
    const depths=[...layers.keys()].sort((a,b)=>a-b),width=Math.max(1,...[...layers.values()].map(l=>l.length))*92+24,height=depths.length*116+24;
    const graph=svg('svg',{width,height,role:'group','aria-label':'Opening DAG'}),positions=new Map();
    depths.forEach((depth,j)=>{const layer=layers.get(depth).sort((a,b)=>a.key.localeCompare(b.key));layer.forEach((n,i)=>positions.set(n.key,[(width-layer.length*92)/2+i*92+12,j*116+12]))});
    for(const key of visible){const n=nodes.get(key),[x,y]=positions.get(key);for(const p of n.parents)if(positions.has(p)){const [px,py]=positions.get(p);graph.append(svg('path',{d:`M${px+32},${py+86} L${x+32},${y}`,stroke:'#898781','stroke-opacity':n.status==='retired'?.2:.5,fill:'none'}))}}
    const detail=element('div');
    const select=async node=>{const id=++this.detailRequest;detail.replaceChildren();try{const result=await this.fetch('/api/book',this.query({run:s.run,key:node.key}));if(id===this.detailRequest&&this.state().view==='dag'&&this.state().run===s.run)detail.append(this.table(result.rows,s))}catch(e){this.error.textContent=e.message}};
    this.detailRequest=0;
    for(const key of visible){const n=nodes.get(key),[x,y]=positions.get(key),g=svg('g',{transform:`translate(${x},${y})`,class:'book-node',tabindex:0,role:'button','aria-label':`${n.status||'prefix'}, depth ${n.depth}, ${n.games} games`});
      const title=svg('title');title.textContent=`${n.status||'prefix'} · depth ${n.depth} · ${n.games} games`;g.append(title);
      g.append(svg('rect',{width:64,height:86,rx:4,fill:'transparent',stroke:statusColour(n.status),class:'node-border'}));const board=miniature(n.moves);board.removeAttribute('tabindex');g.append(board);
      if(n.status==='retired')g.setAttribute('opacity',.4);
      const count=svg('text',{x:32,y:79,fill:statusColour(n.status),'font-size':11,'text-anchor':'middle'});count.textContent=n.games;g.append(count);g.onclick=()=>select(n);g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(n)}};
      if(children.has(key)){const toggle=svg('g',{transform:'translate(68,0)',tabindex:0,role:'button','aria-label':expanded.has(key)?'Collapse descendants':'Expand descendants'});toggle.append(svg('rect',{width:18,height:22,fill:'transparent'}));const text=svg('text',{x:9,y:16,'text-anchor':'middle',fill:statusColour(n.status),'font-size':18});text.textContent=expanded.has(key)?'−':'+';toggle.append(text);const act=e=>{e.stopPropagation();expanded.has(key)?expanded.delete(key):expanded.add(key);this.drawDag(data,s)};toggle.onclick=act;toggle.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();act(e)}};g.append(toggle)}graph.append(g);
    }
    const scroll=element('div',null,{class:'book-dag'});scroll.append(graph);this.content.append(detail,scroll);
  }
}
window.OpeningBook=OpeningBook;
})();
