/* Actor episode lengths, shared by the project and per-run dashboards. */
(() => {
const css=document.createElement('style');
css.textContent=`
.length-panel{background:var(--surface,#fff);border:1px solid var(--ring,#d8e0e5);border-radius:6px;padding:18px;margin-bottom:24px;min-width:0}
.length-head,.length-controls,.length-legend{display:flex;align-items:center;gap:10px 18px;flex-wrap:wrap}.length-head{justify-content:space-between}.length-head h2{margin:0;font-size:16px}
.length-panel label{display:flex;gap:6px;align-items:center;font-size:12px}.length-panel select{font:inherit;color:inherit;background:var(--raised,#f3f5f7);border:1px solid var(--axis,#c5d0d7);border-radius:4px;padding:4px 6px;max-width:180px}
.length-panel p{color:var(--muted,#65757f);font-size:12px;line-height:1.5;margin:12px 0}.length-legend{font-size:12px}.length-legend span{display:flex;align-items:center;gap:6px}.length-legend i{width:9px;height:9px;display:inline-block;border-radius:2px}
.length-chart{display:block;width:100%;height:250px;margin-top:10px;overflow:visible}.length-chart text{fill:var(--muted,#65757f);font:11px system-ui}.length-chart .length-bar{cursor:crosshair}.length-chart .length-bar:hover,.length-chart .length-bar:focus{opacity:.75;outline:none}
.length-controls [hidden]{display:none}.length-note{min-height:36px}.length-panel .length-error{color:#d95926}
`;
document.head.append(css);
const element=(tag,text,attrs={})=>{const e=document.createElement(tag);if(text!=null)e.textContent=text;for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e};
const svg=(tag,attrs={},text)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!=null)e.textContent=text;return e};
const endings=[['win','Played win','#3987e5'],['proven','Proven ending','#199e70'],['capped','Capped / span limit','#c98500']];
const number=v=>v.toLocaleString(undefined,{maximumFractionDigits:1});
window.GameLengths=class {
    constructor(parent,direct=false){
        this.direct=direct;
        this.el=element('section',null,{class:'length-panel'});
        const head=element('div',null,{class:'length-head'}),controls=element('div',null,{class:'length-controls'});
        head.append(element('h2','Self-play game length'),controls);
        const select=(label,options)=>{const sel=element('select',null,{'aria-label':label});for(const [value,text] of options)sel.append(new Option(text,value));const wrap=element('label',label);wrap.append(sel);controls.append(wrap);sel.onchange=()=>this.refresh(true);return sel};
        this.run=select('Run',[]);
        this.hours=select('Window',[[6,'Last 6 hours'],[1,'Last hour'],[24,'Last 24 hours'],[0,'All time']]);
        this.start=select('Starts',[['all','All actor starts'],['selfplay','Normal starts'],['book','Opening book'],['restart','Restart buffer']]);
        this.summary=element('p','Loading game lengths...');this.legend=element('div',null,{class:'length-legend'});
        this.chart=svg('svg',{class:'length-chart',role:'img','aria-label':'Histogram of self-play game lengths in placements'});
        this.hover=element('p','Hover or focus a bar for its game counts.');
        this.el.append(head,this.summary,this.legend,this.chart,this.hover,element('p','Recorded placements include opening and restart prefixes and stored proof continuations. Only published actor shards are counted; windows use shard publication time. Capped games are included.',{class:'length-note'}));
        parent.append(this.el);
        new ResizeObserver(()=>this.draw()).observe(this.chart);
    }
    setRuns(names){
        const signature=names.join('|');
        if(this.signature!==signature){const previous=this.run.value;this.run.replaceChildren(...names.map(n=>new Option(n,n)));if(names.includes(previous))this.run.value=previous;this.signature=signature;this.run.parentElement.hidden=names.length<2}
        this.refresh();
    }
    async refresh(force=false){
        const run=this.run.value;
        if(!run){this.request=(this.request||0)+1;this.key=null;this.data=null;this.summary.textContent='No runs selected';this.draw();return}
        const key=[run,this.hours.value,this.start.value].join('|');
        if(!force&&key===this.key&&Date.now()-this.fetched<30000)return;
        this.key=key;this.fetched=Date.now();const request=this.request=(this.request||0)+1;
        this.data=null;this.draw();
        this.summary.classList.remove('length-error');this.summary.textContent='Loading game lengths...';
        try{
            const query=new URLSearchParams({hours:this.hours.value,start:this.start.value});if(!this.direct)query.set('run',run);
            const res=await fetch('/api/game-lengths?'+query);
            if(!res.ok)throw Error(`Game lengths unavailable: HTTP ${res.status}`);
            const data=await res.json();if(request!==this.request)return;
            this.data=data;this.draw();
        }catch(error){if(request===this.request){this.data=null;this.draw();this.summary.classList.add('length-error');this.summary.textContent=error.message}}
    }
    draw(){
        const data=this.data,w=this.chart.clientWidth,h=250,L=48,R=14,T=24,B=38;
        this.chart.setAttribute('viewBox',`0 0 ${Math.max(1,w)} ${h}`);this.chart.replaceChildren();this.legend.replaceChildren();
        this.hover.textContent='Hover or focus a bar for its game counts.';
        if(!data)return;
        this.summary.textContent=data.games?`${number(data.games)} games · Mean ${number(data.mean)} · Median ${number(data.median)} placements`:'No published actor games in this selection';
        if(!data.games||w<=L+R)return;
        for(const [key,label,color] of endings){const item=element('span');item.append(Object.assign(element('i'),{style:`background:${color}`}),`${label} ${number(data.endings[key]||0)}`);this.legend.append(item)}
        const max=Math.max(...data.bins.map(b=>b.win+b.proven+b.capped)),step=10**Math.floor(Math.log10(max/4||1)),tick=Math.max(1,Math.ceil(max/4/step)*step),top=Math.ceil(max/tick)*tick;
        const dx=(w-L-R)/data.bins.length,y=n=>h-B-n/top*(h-T-B);
        for(let n=0;n<=top;n+=tick){this.chart.append(svg('line',{x1:L,x2:w-R,y1:y(n),y2:y(n),stroke:'var(--grid,#e6ebef)'}),svg('text',{x:L-7,y:y(n)+4,'text-anchor':'end'},number(n)))}
        this.chart.append(svg('text',{x:L,y:12},'Games'),svg('text',{x:w-R,y:h-1,'text-anchor':'end'},'Placements'));
        const labelEvery=Math.max(1,Math.ceil(48/dx));
        data.bins.forEach((b,i)=>{
            if(i%labelEvery===0)this.chart.append(svg('text',{x:L+i*dx,y:h-B+18,'text-anchor':'middle'},b.low));
            const total=b.win+b.proven+b.capped;if(!total)return;
            const text=`${b.low}–${b.high} placements: ${number(total)} games (${(100*total/data.games).toFixed(1)}%). ${endings.map(([key,label])=>`${label}: ${number(b[key])}`).join('; ')}`;
            const group=svg('g',{class:'length-bar',tabindex:0,'aria-label':text});group.append(svg('title',{},text));
            let stacked=0;for(const [key,,color] of endings){if(b[key])group.append(svg('rect',{x:L+i*dx+1,y:y(stacked+b[key]),width:Math.max(1,dx-2),height:y(stacked)-y(stacked+b[key]),fill:color}));stacked+=b[key]}
            group.onpointerenter=group.onfocus=()=>this.hover.textContent=text;
            group.onpointerleave=group.onblur=()=>this.hover.textContent='Hover or focus a bar for its game counts.';
            this.chart.append(group);
        });
    }
};
})();
