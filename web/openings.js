let openingData=[], openingSignature='';
function renderOpenings(openings){
  $('opening-section').hidden=false;
  const next=[{id:'selfplay',moves:[],games:[]},...openings],signature=JSON.stringify(next);
  if(signature!==openingSignature){
    const selected=$('opening-select').value;openingData=next;openingSignature=signature;
    $('opening-select').replaceChildren();
    for(const item of next){
      const option=document.createElement('option');option.value=item.id;
      option.textContent=item.id==='selfplay'?'Self-play: empty board':`Checkpoint ${item.candidate} vs ${item.opponent}, pair ${item.pair+1}`;
      $('opening-select').append(option);
    }
    $('opening-select').value=next.some(o=>o.id===selected)?selected:(next[1]?.id||'selfplay');
    chooseOpening();
  }else drawOpening();
}
function chooseOpening(){
  const item=openingData.find(o=>o.id===$('opening-select').value);if(!item)return;
  $('opening-step').max=item.moves.length;$('opening-step').value=item.moves.length;
  $('opening-step').disabled=!item.moves.length;drawOpening();
}
function drawOpening(){
  const item=openingData.find(o=>o.id===$('opening-select').value);if(!item)return;
  const step=Number($('opening-step').value),moves=item.moves.slice(0,step),canvas=$('opening-board'),ratio=devicePixelRatio||1,w=canvas.clientWidth,h=320;
  canvas.width=Math.round(w*ratio);canvas.height=h*ratio;const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
  const points=item.moves.length?item.moves:[[0,0]],margin=2;
  const qmin=Math.min(...points.map(p=>p[0]))-margin,qmax=Math.max(...points.map(p=>p[0]))+margin;
  const rmin=Math.min(...points.map(p=>p[1]))-margin,rmax=Math.max(...points.map(p=>p[1]))+margin;
  const cells=[];for(let q=qmin;q<=qmax;q++)for(let r=rmin;r<=rmax;r++)cells.push([q,r,Math.sqrt(3)*(q+r/2),-1.5*r]);
  const xs=cells.map(p=>p[2]),ys=cells.map(p=>p[3]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const size=Math.min(29,(w-20)/(xmax-xmin+2),(h-20)/(ymax-ymin+2));
  for(const [q,r,px,py] of cells){
    const x=w/2+(px-(xmin+xmax)/2)*size,y=h/2+(py-(ymin+ymax)/2)*size,index=moves.findIndex(p=>p[0]===q&&p[1]===r);
    ctx.beginPath();
    for(let k=0;k<6;k++){const a=(60*k-30)*Math.PI/180,xx=x+size*.94*Math.cos(a),yy=y+size*.94*Math.sin(a);k?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy)}
    ctx.closePath();ctx.fillStyle=index<0?'#f7f9fa':index===0?'#175e83':'#b9634d';ctx.fill();ctx.strokeStyle='#dbe3e8';ctx.stroke();
    if(index>=0||q===0&&r===0){ctx.fillStyle=index>=0?'white':'#65757f';ctx.font=`${Math.max(10,size*.6)}px system-ui`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(index>=0?String(index+1):'0,0',x,y)}
  }
  $('opening-position').textContent=` ${step} / ${item.moves.length}`;
  $('opening-detail').textContent=item.id==='selfplay'?'Player 1 places the first stone at (0, 0). No stones are pre-filled.':`Saved games ${item.games.map(i=>i+1).join(' and ')}. Seed ${item.seed}. After stone 3, Player 1 has two placements. Blue is Player 1; red is Player 2.`;
  $('opening-sequence').textContent=item.moves.length?item.moves.map((p,i)=>`${i+1}. Player ${i===0?1:2}: (${p[0]}, ${p[1]})`).join(' → '):'Empty board → Player 1: (0, 0) → Player 2 places two stones.';
}
document.getElementById('opening-select').onchange=chooseOpening;
document.getElementById('opening-step').oninput=drawOpening;
