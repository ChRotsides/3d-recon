"""Write a self-contained HTML page comparing VIO trajectories with the COLMAP reference."""
import sys, json
import numpy as np
from pathlib import Path
out = Path(sys.argv[1]); runs = {}
for spec in sys.argv[2:]:
    label, path = spec.split("=")
    d = np.load(path)
    runs[label] = {"vio": d["vio"].round(4).tolist(), "vio_se3": d["vio_se3"].round(4).tolist(), "ref": d["ref"].round(4).tolist(), "t": (d["t"] - d["t"][0]).round(3).tolist()}
html = r'''<!doctype html><html><head><meta charset="utf-8"><title>VIO vs COLMAP</title>
<style>body{font-family:system-ui;margin:0;background:#111;color:#ddd}#top{padding:8px 12px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
canvas{background:#181818;display:block}#wrap{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px}
label{margin-right:8px}.k{display:inline-block;width:14px;height:3px;margin-right:4px;vertical-align:middle}
#stats{font-family:monospace;white-space:pre;padding:0 12px 12px}</style></head><body>
<div id="top"><b>VIO vs COLMAP, walk rec_20260903-070045</b>
<span>run: <select id="run"></select></span>
<span><i class="k" style="background:#4af"></i>COLMAP + floor scale (reference)</span>
<span><i class="k" style="background:#fa4"></i>VIO</span> <span>alignment: <select id="align"><option value="vio">Sim(3): scale fitted</option><option value="vio_se3">SE(3): VIO metric scale kept</option></select></span>
<span><input type="range" id="time" min="0" max="1000" value="1000" style="width:260px"> <span id="tlabel"></span></span>
</div>
<div id="wrap"><div><div style="padding:4px 8px">Top-down (X–Z, metres)</div><canvas id="topview"></canvas></div>
<div><div style="padding:4px 8px">Side (X–Y height, metres)</div><canvas id="side"></canvas></div>
<div><div style="padding:4px 8px">Position error over time (cm)</div><canvas id="err"></canvas></div>
<div><div style="padding:4px 8px">Height over time (m)</div><canvas id="height"></canvas></div></div>
<div id="stats"></div>
<script>
const RUNS = __RUNS__;
const sel=document.getElementById('run'); for(const k in RUNS){const o=document.createElement('option');o.value=k;o.textContent=k;sel.appendChild(o)}
const slider=document.getElementById('time');
function umeyama(A,B){const n=A.length;const ma=[0,0,0],mb=[0,0,0];for(const p of A)for(let i=0;i<3;i++)ma[i]+=p[i]/n;for(const p of B)for(let i=0;i<3;i++)mb[i]+=p[i]/n;
// use numeric.js-free 3x3 SVD via Jacobi on H^T H; fine for small demo -> instead precomputed alignment is done in python, so here just identity
return null}
function draw(){
 const r0=RUNS[sel.value]; const r={...r0, vio:r0[document.getElementById('align').value]}; const n=r.vio.length; const upto=Math.max(2,Math.round(n*slider.value/1000));
 document.getElementById('tlabel').textContent=r.t[upto-1].toFixed(1)+' s / '+r.t[n-1].toFixed(1)+' s';
 const ref=r.ref.slice(0,upto), vio=r.vio.slice(0,upto);
 plotXY('topview',ref,vio,0,2,true); plotXY('side',ref,vio,0,1,true);
 const err=r.ref.map((p,i)=>100*Math.hypot(p[0]-r.vio[i][0],p[1]-r.vio[i][1],p[2]-r.vio[i][2]));
 series('err',r.t,[{v:err,c:'#fa4'}],upto,0);
 series('height',r.t,[{v:r.ref.map(p=>p[1]),c:'#4af'},{v:r.vio.map(p=>p[1]),c:'#fa4'}],upto,null);
 const e=err.slice(0,upto); const rmse=Math.sqrt(e.reduce((a,b)=>a+b*b,0)/e.length); const med=[...e].sort((a,b)=>a-b)[Math.floor(e.length/2)];
 document.getElementById('stats').textContent=`keyframes shown ${upto}/${n}   position error: rmse ${rmse.toFixed(1)} cm  median ${med.toFixed(1)} cm  max ${Math.max(...e).toFixed(1)} cm`;
}
function fit(id){const c=document.getElementById(id);const w=c.parentElement.clientWidth-0;c.width=w;c.height=Math.round(w*0.6);return c}
function plotXY(id,ref,vio,ax,ay,equal){const c=fit(id),g=c.getContext('2d');const all=ref.concat(vio);
 let x0=Math.min(...all.map(p=>p[ax])),x1=Math.max(...all.map(p=>p[ax])),y0=Math.min(...all.map(p=>p[ay])),y1=Math.max(...all.map(p=>p[ay]));
 const pad=0.5;x0-=pad;x1+=pad;y0-=pad;y1+=pad;const sx=c.width/(x1-x0),sy=c.height/(y1-y0);const s=Math.min(sx,sy);
 const X=v=>(v-x0)*s+ (c.width-(x1-x0)*s)/2, Y=v=>c.height-((v-y0)*s+(c.height-(y1-y0)*s)/2);
 g.strokeStyle='#333';g.lineWidth=1;for(let m=Math.ceil(x0);m<=x1;m++){g.beginPath();g.moveTo(X(m),0);g.lineTo(X(m),c.height);g.stroke()}
 for(let m=Math.ceil(y0);m<=y1;m++){g.beginPath();g.moveTo(0,Y(m));g.lineTo(c.width,Y(m));g.stroke()}
 g.fillStyle='#666';g.font='11px monospace';g.fillText('1 m grid',6,12);
 for(const [pts,col] of [[ref,'#4af'],[vio,'#fa4']]){g.strokeStyle=col;g.lineWidth=2;g.beginPath();pts.forEach((p,i)=>{i?g.lineTo(X(p[ax]),Y(p[ay])):g.moveTo(X(p[ax]),Y(p[ay]))});g.stroke();
  const l=pts[pts.length-1];g.fillStyle=col;g.beginPath();g.arc(X(l[ax]),Y(l[ay]),4,0,7);g.fill()}
 g.strokeStyle='#555';g.lineWidth=1;for(let i=0;i<ref.length;i+=6){g.beginPath();g.moveTo(X(ref[i][ax]),Y(ref[i][ay]));g.lineTo(X(vio[i][ax]),Y(vio[i][ay]));g.stroke()}
}
function series(id,t,ss,upto,ymin){const c=fit(id),g=c.getContext('2d');const t1=t[t.length-1];let lo=ymin,hi=-1e9;
 for(const s of ss){for(const v of s.v){if(ymin===null)lo=Math.min(lo??1e9,v);hi=Math.max(hi,v)}}if(lo===null||lo===undefined)lo=Math.min(...ss[0].v);const pad=(hi-lo)*0.1+1e-6;lo-=pad;hi+=pad;
 const X=v=>v/t1*c.width,Y=v=>c.height-(v-lo)/(hi-lo)*c.height;
 g.strokeStyle='#333';for(let k=0;k<=4;k++){const y=lo+(hi-lo)*k/4;g.beginPath();g.moveTo(0,Y(y));g.lineTo(c.width,Y(y));g.stroke();g.fillStyle='#777';g.font='11px monospace';g.fillText(y.toFixed(1),4,Y(y)-2)}
 for(const s of ss){g.strokeStyle=s.c;g.lineWidth=1.5;g.beginPath();for(let i=0;i<upto;i++){i?g.lineTo(X(t[i]),Y(s.v[i])):g.moveTo(X(t[i]),Y(s.v[i]))}g.stroke()}
 g.strokeStyle='#eee';g.beginPath();g.moveTo(X(t[upto-1]),0);g.lineTo(X(t[upto-1]),c.height);g.stroke();
}
sel.onchange=draw;document.getElementById('align').onchange=draw;slider.oninput=draw;window.onresize=draw;draw();
</script></body></html>'''
out.write_text(html.replace("__RUNS__", json.dumps(runs)))
print("wrote", out, {k: len(v["t"]) for k, v in runs.items()})
