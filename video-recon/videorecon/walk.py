# ruff: noqa: E501
"""Synchronised playback: the video on one side, the fused cloud on the other.

The video's clock is the master. Every keyframe in the bundle has a video frame
index (its id) and therefore a presentation time, and a pose in the repo frame;
the page slerps between keyframes at the video's `currentTime` and renders the
cloud from that pose. The sensor-native frames are landscape, so both the
preview video and the virtual camera are rotated by the sensor orientation to
show the room upright.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import threading
from pathlib import Path

import numpy as np

from videorecon.recording import video_frame_pts_ns

_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}


def recording_dir(root: Path) -> Path | None:
    manifest = root / "manifest.json"
    if manifest.exists():
        try:
            rec = Path(json.loads(manifest.read_text())["recording"])
            if rec.exists():
                return rec
        except (KeyError, json.JSONDecodeError):
            pass
    prov = root / "bundle" / "provenance.json"
    if prov.exists():
        try:
            rec = Path(json.loads(prov.read_text())["manifest_stages"]["bundle"]["path"]).parents[2]
        except (KeyError, json.JSONDecodeError):
            rec = None
    return None


def sensor_orientation(root: Path) -> int:
    prov = root / "bundle" / "provenance.json"
    if prov.exists():
        try:
            return (
                int(
                    json.loads(prov.read_text())["manifest_stages"]["recording"][
                        "sensor_orientation_deg"
                    ]
                )
                % 360
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return 90


def camera_rotation(root: Path) -> int:
    """Rotation still needed to show the bundle's camera upright: 0 once the
    bundle itself was rotated upright for the depth model."""
    prov = root / "bundle" / "provenance.json"
    if prov.exists():
        with contextlib.suppress(KeyError, ValueError, json.JSONDecodeError):
            stages = json.loads(prov.read_text())["manifest_stages"]
            if "bundle_rotation_deg" in stages:
                return (sensor_orientation(root) - int(stages["bundle_rotation_deg"])) % 360
    return sensor_orientation(root)


def preview_status(root: Path, recording: Path | None) -> dict:
    """Kick off (once) a small upright, fast-start transcode for the browser."""
    out = root / "preview.mp4"
    if out.exists():
        return {"ready": True, "url": "/out/preview.mp4"}
    if recording is None or not (recording / "video.mp4").exists():
        return {"ready": False, "error": "recording video not found"}
    key = str(root)
    with _LOCK:
        job = _JOBS.get(key)
        if job is None:
            job = {"state": "running", "error": None}
            _JOBS[key] = job
            threading.Thread(
                target=_transcode,
                args=(recording / "video.mp4", out, sensor_orientation(root), job),
                daemon=True,
            ).start()
    return {"ready": False, "state": job["state"], "error": job["error"]}


def _transcode(src: Path, dst: Path, rotation: int, job: dict) -> None:
    transpose = {90: "transpose=1,", 270: "transpose=2,", 180: "transpose=1,transpose=1,"}.get(
        rotation, ""
    )
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        "-vf",
        f"{transpose}scale=-2:960",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "25",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        tmp.replace(dst)
        job["state"] = "done"
    except subprocess.CalledProcessError as exc:
        job["state"] = "failed"
        job["error"] = exc.stderr[-400:]


def trajectory(root: Path, recording: Path | None) -> dict:
    """Keyframe poses with their video times, ready for the page."""
    bundle = root / "bundle" / "bundle.json"
    if not bundle.exists():
        return {"error": "no bundle yet"}
    b = json.loads(bundle.read_text())
    kf_json = root / "keyframes.json"
    fps = 30.0
    pts = None
    if recording is not None and (recording / "video.mp4").exists():
        cache = root / "video_pts.json"
        if cache.exists():
            pts = np.asarray(json.loads(cache.read_text()), dtype=np.int64)
        else:
            pts = video_frame_pts_ns(recording / "video.mp4")
            cache.write_text(json.dumps(pts.tolist()))
        with contextlib.suppress(KeyError, json.JSONDecodeError, FileNotFoundError):
            fps = float(json.loads((recording / "meta.json").read_text())["video"]["fps"])
    frames = []
    for k in b["keyframes"]:
        idx = int(k["id"][2:]) if k["id"].startswith("kf") else None
        if idx is None:
            continue
        t = float(pts[idx]) * 1e-9 if pts is not None and idx < len(pts) else idx / fps
        frames.append({"i": idx, "t": t, "R": k["pose"]["R"], "p": k["pose"]["t"]})
    frames.sort(key=lambda f: f["t"])
    intr = b["keyframes"][0]["intrinsics"] if b["keyframes"] else {}
    return {
        "frames": frames,
        "sensor_orientation_deg": camera_rotation(root),
        "intrinsics": intr,
        "keyframes_json": kf_json.exists(),
    }


WALK_PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Walk the reconstruction</title>
<style>
html,body{margin:0;height:100%;background:#0d0d0d;color:#ddd;font-family:system-ui,sans-serif;overflow:hidden}
#wrap{display:flex;height:100vh}
#left{width:380px;min-width:300px;display:flex;flex-direction:column;background:#151515;border-right:1px solid #222}
video{width:100%;background:#000;flex:0 0 auto;max-height:70vh}
#info{padding:10px 12px;font-size:12px;color:#aaa;line-height:1.6}
#right{flex:1;position:relative}
canvas{width:100%;height:100%;display:block}
#hud{position:absolute;top:10px;left:12px;background:#0009;padding:8px 10px;border-radius:6px;font-size:12px}
#hud select,#hud input,#hud button{margin-left:6px}
a{color:#80cbc4}
.warn{color:#ff8a65}
</style></head><body>
<div id="wrap">
 <div id="left">
  <video id="video" controls preload="auto" playsinline></video>
  <div id="info">
   <div id="vstatus">preparing preview video…</div>
   <div>time <span id="time">0.00</span> s · keyframe <span id="kf">–</span> · <span id="ptsinfo"></span></div>
   <div>Space play/pause · ←/→ step 1 s · drag the 3D view to orbit (orbit mode) · <a href="/">dashboard</a></div>
  </div>
 </div>
 <div id="right">
  <canvas id="gl"></canvas>
  <div id="hud">view <select id="mode"><option value="first">first person (what the phone saw)</option><option value="follow">follow (behind the phone)</option><option value="orbit">orbit (whole room, path drawn)</option></select>
   point <input id="psize" type="range" min="1" max="8" value="2" style="width:80px"> <span id="cloudinfo"></span></div>
 </div>
</div>
<script>
const $=id=>document.getElementById(id);
const video=$('video');let traj=null,cloud=null,gl=null,prog=null,lineProg=null,buffers={},N=0,rotDeg=90,fovY=1.0;
let orbit={yaw:0.6,pitch:0.5,dist:8,target:[0,0,0],drag:false,lx:0,ly:0};
// ---------- small matrix helpers (column-major, WebGL) ----------
function perspective(fovy,aspect,n,f){const t=1/Math.tan(fovy/2);return new Float32Array([t/aspect,0,0,0,0,t,0,0,0,0,(f+n)/(n-f),-1,0,0,2*f*n/(n-f),0]);}
function mul(a,b){const o=new Float32Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;}return o;}
function viewFromAxes(right,up,back,eye){ // world axes of the camera; view = inverse of [R|eye]
 const r=right,u=up,b=back;return new Float32Array([r[0],u[0],b[0],0, r[1],u[1],b[1],0, r[2],u[2],b[2],0, -(r[0]*eye[0]+r[1]*eye[1]+r[2]*eye[2]), -(u[0]*eye[0]+u[1]*eye[1]+u[2]*eye[2]), -(b[0]*eye[0]+b[1]*eye[1]+b[2]*eye[2]),1]);}
function lookAt(eye,target,upHint){const b=norm(sub(eye,target));let r=norm(cross(upHint,b));if(!isFinite(r[0]))r=[1,0,0];const u=cross(b,r);return viewFromAxes(r,u,b,eye);}
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]],scl=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm=a=>{const l=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/l,a[1]/l,a[2]/l];};
// rotation matrices (row-major 3x3 from JSON) <-> quaternions
function m2q(m){const t=m[0][0]+m[1][1]+m[2][2];let q;if(t>0){const s=Math.sqrt(t+1)*2;q=[(m[2][1]-m[1][2])/s,(m[0][2]-m[2][0])/s,(m[1][0]-m[0][1])/s,0.25*s];}
 else if(m[0][0]>m[1][1]&&m[0][0]>m[2][2]){const s=Math.sqrt(1+m[0][0]-m[1][1]-m[2][2])*2;q=[0.25*s,(m[0][1]+m[1][0])/s,(m[0][2]+m[2][0])/s,(m[2][1]-m[1][2])/s];}
 else if(m[1][1]>m[2][2]){const s=Math.sqrt(1+m[1][1]-m[0][0]-m[2][2])*2;q=[(m[0][1]+m[1][0])/s,0.25*s,(m[1][2]+m[2][1])/s,(m[0][2]-m[2][0])/s];}
 else{const s=Math.sqrt(1+m[2][2]-m[0][0]-m[1][1])*2;q=[(m[0][2]+m[2][0])/s,(m[1][2]+m[2][1])/s,0.25*s,(m[1][0]-m[0][1])/s];}return q;}
function q2m(q){const[x,y,z,w]=q;return[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];}
function slerp(a,b,t){let d=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3];if(d<0){b=b.map(v=>-v);d=-d;}if(d>0.9995){const o=a.map((v,i)=>v+(b[i]-v)*t);const l=Math.hypot(...o);return o.map(v=>v/l);}
 const th=Math.acos(d),s=Math.sin(th);return a.map((v,i)=>(v*Math.sin((1-t)*th)+b[i]*Math.sin(t*th))/s);}
// pose at video time: slerp between keyframes
function poseAt(t){const f=traj.frames;if(!f.length)return null;if(t<=f[0].t)return{R:f[0].R,p:f[0].p,i:f[0].i};if(t>=f[f.length-1].t){const l=f[f.length-1];return{R:l.R,p:l.p,i:l.i};}
 let lo=0,hi=f.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(f[m].t<=t)lo=m;else hi=m;}
 const a=f[lo],b=f[hi],u=(t-a.t)/Math.max(b.t-a.t,1e-6);const q=slerp(a.q,b.q,u);return{R:q2m(q),p:[a.p[0]+(b.p[0]-a.p[0])*u,a.p[1]+(b.p[1]-a.p[1])*u,a.p[2]+(b.p[2]-a.p[2])*u],i:a.i};}
// camera axes in world for an upright display: rotate image axes by the sensor orientation
function uprightAxes(R){const x=[R[0][0],R[1][0],R[2][0]],y=[R[0][1],R[1][1],R[2][1]],z=[R[0][2],R[1][2],R[2][2]];const th=rotDeg*Math.PI/180,c=Math.cos(th),s=Math.sin(th);
 const right=add(scl(x,c),scl(y,s)),up=add(scl(x,-s),scl(y,c));return{right,up,back:z,forward:scl(z,-1)};}
// ---------- WebGL ----------
const VS=`#version 300 es
layout(location=0) in vec3 pos;layout(location=1) in vec3 col;uniform mat4 mvp;uniform float psize;out vec3 vc;
void main(){gl_Position=mvp*vec4(pos,1.0);float w=max(gl_Position.w,0.05);gl_PointSize=clamp(psize*2.0/w,1.0,10.0);vc=col;}`;
const FS=`#version 300 es
precision mediump float;in vec3 vc;out vec4 o;void main(){o=vec4(vc,1.0);}`;
const LVS=`#version 300 es
layout(location=0) in vec3 pos;uniform mat4 mvp;void main(){gl_Position=mvp*vec4(pos,1.0);}`;
const LFS=`#version 300 es
precision mediump float;uniform vec3 color;out vec4 o;void main(){o=vec4(color,1.0);}`;
function shader(t,s){const sh=gl.createShader(t);gl.shaderSource(sh,s);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(sh);return sh;}
function program(v,f){const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,v));gl.attachShader(p,shader(gl.FRAGMENT_SHADER,f));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw gl.getProgramInfoLog(p);return p;}
function initGL(){const c=$('gl');gl=c.getContext('webgl2');if(!gl){$('cloudinfo').textContent='WebGL2 unavailable';return;}prog=program(VS,FS);lineProg=program(LVS,LFS);gl.enable(gl.DEPTH_TEST);gl.clearColor(0.05,0.05,0.06,1);}
async function loadCloud(){const meta=await(await fetch('/out/cloud.json?t='+Date.now())).json();const buf=await(await fetch('/out/cloud.bin?t='+Date.now())).arrayBuffer();N=meta.count;
 const pos=new Float32Array(buf,0,N*3),col=new Uint8Array(buf,N*12,N*3);
 buffers.pos=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffers.pos);gl.bufferData(gl.ARRAY_BUFFER,pos,gl.STATIC_DRAW);
 buffers.col=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffers.col);gl.bufferData(gl.ARRAY_BUFFER,col,gl.STATIC_DRAW);
 orbit.target=meta.view_centre||meta.centre;orbit.dist=Math.max(...(meta.view_extent||meta.extent))*1.2;cloud=meta;$('cloudinfo').textContent=`${N.toLocaleString()} points${meta.partial?' (partial)':''}`;}
function loadPath(){const f=traj.frames;const a=new Float32Array(f.length*3);f.forEach((k,i)=>{a[i*3]=k.p[0];a[i*3+1]=k.p[1];a[i*3+2]=k.p[2];});buffers.path=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffers.path);gl.bufferData(gl.ARRAY_BUFFER,a,gl.STATIC_DRAW);buffers.pathN=f.length;buffers.frustum=gl.createBuffer();}
function drawLines(buf,n,mvp,color,mode){gl.useProgram(lineProg);gl.uniformMatrix4fv(gl.getUniformLocation(lineProg,'mvp'),false,mvp);gl.uniform3fv(gl.getUniformLocation(lineProg,'color'),color);gl.bindBuffer(gl.ARRAY_BUFFER,buf);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);gl.disableVertexAttribArray(1);gl.drawArrays(mode,0,n);}
function frustumLines(pose){const ax=uprightAxes(pose.R),e=pose.p,d=0.35,h=Math.tan(fovY/2)*d,w=h*9/16;const c=add(e,scl(ax.forward,d));
 const corners=[add(add(c,scl(ax.right,w)),scl(ax.up,h)),add(add(c,scl(ax.right,-w)),scl(ax.up,h)),add(add(c,scl(ax.right,-w)),scl(ax.up,-h)),add(add(c,scl(ax.right,w)),scl(ax.up,-h))];
 const L=[];for(let i=0;i<4;i++){L.push(...e,...corners[i]);L.push(...corners[i],...corners[(i+1)%4]);}return new Float32Array(L);}
function render(){if(!gl||!cloud||!traj)return requestAnimationFrame(render);const c=$('gl');const W=c.clientWidth,H=c.clientHeight;if(c.width!==W||c.height!==H){c.width=W;c.height=H;gl.viewport(0,0,W,H);}
 const t=video.currentTime||0;const pose=poseAt(t);$('time').textContent=t.toFixed(2);$('kf').textContent=pose?pose.i:'–';
 const mode=$('mode').value;let view,proj;const aspect=W/H;
 if(pose&&mode==='first'){const ax=uprightAxes(pose.R);view=viewFromAxes(ax.right,ax.up,ax.back,pose.p);proj=perspective(fovY,aspect,0.05,60);}
 else if(pose&&mode==='follow'){const ax=uprightAxes(pose.R);const eye=add(add(pose.p,scl(ax.forward,-1.6)),[0,0.9,0]);view=lookAt(eye,add(pose.p,scl(ax.forward,1.5)),[0,1,0]);proj=perspective(1.1,aspect,0.05,60);}
 else{const cy=Math.cos(orbit.pitch),sy=Math.sin(orbit.pitch);const eye=add(orbit.target,[orbit.dist*cy*Math.sin(orbit.yaw),orbit.dist*sy,orbit.dist*cy*Math.cos(orbit.yaw)]);view=lookAt(eye,orbit.target,[0,1,0]);proj=perspective(1.0,aspect,0.05,200);}
 const mvp=mul(proj,view);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 gl.useProgram(prog);gl.uniformMatrix4fv(gl.getUniformLocation(prog,'mvp'),false,mvp);gl.uniform1f(gl.getUniformLocation(prog,'psize'),+$('psize').value);
 gl.bindBuffer(gl.ARRAY_BUFFER,buffers.pos);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
 gl.bindBuffer(gl.ARRAY_BUFFER,buffers.col);gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,3,gl.UNSIGNED_BYTE,true,0,0);gl.drawArrays(gl.POINTS,0,N);
 if(mode!=='first'&&pose){drawLines(buffers.path,buffers.pathN,mvp,[1,0.3,0.2],gl.LINE_STRIP);const fl=frustumLines(pose);gl.bindBuffer(gl.ARRAY_BUFFER,buffers.frustum);gl.bufferData(gl.ARRAY_BUFFER,fl,gl.DYNAMIC_DRAW);drawLines(buffers.frustum,fl.length/3,mvp,[1,0.95,0.3],gl.LINES);}
 requestAnimationFrame(render);}
// ---------- input ----------
const cv=$('gl');cv.addEventListener('mousedown',e=>{orbit.drag=true;orbit.lx=e.clientX;orbit.ly=e.clientY;});window.addEventListener('mouseup',()=>orbit.drag=false);
window.addEventListener('mousemove',e=>{if(!orbit.drag)return;orbit.yaw-=(e.clientX-orbit.lx)*0.006;orbit.pitch=Math.max(-1.4,Math.min(1.4,orbit.pitch+(e.clientY-orbit.ly)*0.006));orbit.lx=e.clientX;orbit.ly=e.clientY;if($('mode').value!=='orbit')$('mode').value='orbit';});
cv.addEventListener('wheel',e=>{orbit.dist*=Math.exp(e.deltaY*0.001);e.preventDefault();},{passive:false});
window.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;if(e.code==='Space'){e.preventDefault();video.paused?video.play():video.pause();}if(e.code==='ArrowRight')video.currentTime+=1;if(e.code==='ArrowLeft')video.currentTime=Math.max(0,video.currentTime-1);});
// ---------- boot ----------
async function boot(){initGL();traj=await(await fetch('/walk/trajectory')).json();if(traj.error){$('vstatus').textContent=traj.error;return;}
 traj.frames.forEach(f=>f.q=m2q(f.R));rotDeg=traj.sensor_orientation_deg||90;const K=traj.intrinsics;
 if(K&&K.fx){const upright=(rotDeg%180)!==0;const H=upright?K.width:K.height;fovY=2*Math.atan(H/2/K.fy);} 
 $('ptsinfo').textContent=`${traj.frames.length} keyframes, ${traj.frames[traj.frames.length-1].t.toFixed(1)} s of path`;
 loadPath();await loadCloud();render();
 const poll=async()=>{const s=await(await fetch('/walk/status')).json();if(s.ready){video.src=s.url;$('vstatus').textContent='video ready — press play';}else{$('vstatus').textContent=s.error?('preview failed: '+s.error):'preparing an upright preview video (one-off, about a minute)…';setTimeout(poll,3000);}};poll();}
boot();
</script></body></html>"""
