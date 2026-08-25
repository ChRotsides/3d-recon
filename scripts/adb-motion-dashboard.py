#!/usr/bin/env python3
"""Stream motion sensor values exposed by an Android device over wireless ADB.

This deliberately does not install or launch anything on the phone. Android's
sensorservice keeps a recent-event ring, which this script polls and republishes
as a local Server-Sent Events stream.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


MOTION_TYPES = {
    1, 2, 3, 4, 9, 10, 11, 14, 15, 16, 17, 18, 19, 22, 25, 27, 30, 35
}
MOTION_NAME = re.compile(
    r"acceler|gyro|magnet|gravity|linear.?accel|rotation|orientation|"
    r"\bstep\b|motion|tilt|gesture|hinge|pose|drop|grip|stationary|pocket",
    re.IGNORECASE,
)
META_RE = re.compile(
    r"^\s*(0x[0-9a-fA-F]+)\)\s*(.*?)\s*\|\s*(.*?)\s*\|\s*ver:\s*(.*?)"
    r"\s*\|\s*type:\s*(.*?)\((\d+)\)\s*\|"
)
EVENT_HEADER_RE = re.compile(r"^(.+): last \d+ events$")
EVENT_RE = re.compile(
    r"^\s*\d+\s+\(ts=([^,]+),\s*wall=([^)]+)\)\s*(.*)$"
)


def find_adb(explicit: str | None) -> str:
    candidates = [
        explicit,
        os.environ.get("ADB"),
        shutil.which("adb"),
        os.path.join(os.environ.get("ANDROID_HOME", ""), "platform-tools", "adb"),
        os.path.join(os.environ.get("ANDROID_SDK_ROOT", ""), "platform-tools", "adb"),
        "/home/charalambos/Android/Sdk/platform-tools/adb",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise SystemExit(
        "adb was not found. Install platform-tools, set ADB=/path/to/adb, "
        "or pass --adb /path/to/adb."
    )


def run_adb(adb: str, serial: str, *args: str, timeout: float = 12.0) -> str:
    command = [adb, "-s", serial, *args]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "adb command failed")
    return result.stdout


def choose_device(adb: str, requested: str | None) -> str:
    serial = requested or os.environ.get("ANDROID_SERIAL")
    if serial:
        return serial
    result = subprocess.run(
        [adb, "devices"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    if result.returncode:
        raise SystemExit(result.stderr.strip() or "adb devices failed")
    devices = [
        line.split()[0]
        for line in result.stdout.splitlines()[1:]
        if len(line.split()) >= 2 and line.split()[1] == "device"
    ]
    if not devices:
        raise SystemExit(
            "No authorized ADB device found. Enable Wireless debugging, pair/connect "
            "the phone, then check with: adb devices"
        )
    if len(devices) > 1:
        print("Multiple devices; using " + devices[0], file=sys.stderr)
    return devices[0]


def is_motion(meta: dict[str, Any]) -> bool:
    return int(meta["type"]) in MOTION_TYPES or bool(MOTION_NAME.search(meta["name"]))


def parse_metadata(text: str) -> dict[str, dict[str, Any]]:
    start = text.find("Sensor List:")
    end = text.find("Fusion States:", start)
    if start < 0:
        return {}
    if end < 0:
        end = len(text)
    sensors: dict[str, dict[str, Any]] = {}
    for line in text[start:end].splitlines():
        match = META_RE.match(line)
        if not match:
            continue
        handle, name, vendor, version, type_name, type_number = match.groups()
        item = {
            "sensorId": handle,
            "name": name.strip(),
            "vendor": vendor.strip(),
            "version": version.strip(),
            "type": int(type_number),
            "typeName": type_name.strip() or "custom",
            "descriptor": line.strip(),
        }
        if is_motion(item):
            sensors[handle] = item
    return sensors


def parse_value_tokens(raw: str) -> tuple[list[Any], list[str]]:
    raw = raw.strip().rstrip(",").strip()
    if not raw or raw == "[value masked]":
        return [], [raw] if raw else []
    tokens = [token.strip() for token in raw.split(",")]
    values: list[Any] = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            values.append(token)
    return values, tokens


def parse_events(text: str, sensors: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    start = text.find("Recent Sensor events:")
    end = text.find("Active sensors:", start)
    if start < 0:
        return []
    if end < 0:
        end = len(text)
    current_sensor_name: str | None = None
    by_name = {item["name"]: item for item in sensors.values()}
    events: list[dict[str, Any]] = []
    for line in text[start:end].splitlines():
        header = EVENT_HEADER_RE.match(line)
        if header:
            current_sensor_name = header.group(1).strip()
            continue
        match = EVENT_RE.match(line)
        if not match or current_sensor_name is None:
            continue
        meta = by_name.get(current_sensor_name)
        if meta is None:
            continue
        timestamp, wall, raw = match.groups()
        values, raw_values = parse_value_tokens(raw)
        events.append({
            "sensorId": meta["sensorId"],
            "name": meta["name"],
            "type": meta["type"],
            "typeName": meta["typeName"],
            "timestamp": timestamp.strip(),
            "wallTime": wall.strip(),
            "values": values,
            "rawValues": raw_values,
        })
    return events
class EventHub:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sensors: dict[str, dict[str, Any]] = {}
        self.latest: dict[str, dict[str, Any]] = {}
        self.history: collections.deque[dict[str, Any]] = collections.deque(maxlen=4000)
        self.clients: set[queue.Queue[str]] = set()

    @staticmethod
    def event_message(event: dict[str, Any]) -> str:
        return "data: " + json.dumps(event, separators=(",", ":")) + "\n\n"

    def update(self, sensors: dict[str, dict[str, Any]], events: list[dict[str, Any]]) -> None:
        with self.lock:
            self.sensors.update(sensors)
            for event in events:
                self.history.append(event)
                self.latest[event["sensorId"]] = event
                message = self.event_message(event)
                for client in list(self.clients):
                    try:
                        client.put_nowait(message)
                    except queue.Full:
                        try:
                            client.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            client.put_nowait(message)
                        except queue.Full:
                            pass

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "serverTimeMs": int(time.time() * 1000),
                "sensors": list(self.sensors.values()),
                "latest": list(self.latest.values()),
            }

    def subscribe(self) -> queue.Queue[str]:
        # Replay the recent sensorservice ring as individual events. The 3D
        # view needs the sequence, not only the latest value per sensor.
        client: queue.Queue[str] = queue.Queue(maxsize=8192)
        with self.lock:
            self.clients.add(client)
            snapshot = {
                "serverTimeMs": int(time.time() * 1000),
                "sensors": list(self.sensors.values()),
                "latest": list(self.latest.values()),
            }
            history = list(self.history)
            client.put_nowait("data: " + json.dumps(snapshot, separators=(",", ":")) + "\n\n")
            for event in history:
                client.put_nowait(self.event_message(event))
        return client

    def unsubscribe(self, client: queue.Queue[str]) -> None:
        with self.lock:
            self.clients.discard(client)


class Poller(threading.Thread):
    def __init__(self, adb: str, serial: str, hub: EventHub, interval: float) -> None:
        super().__init__(daemon=True)
        self.adb = adb
        self.serial = serial
        self.hub = hub
        self.interval = interval
        self.stop_event = threading.Event()
        self.seen: set[tuple[str, str, str]] = set()
        self.seen_order: collections.deque[tuple[str, str, str]] = collections.deque()
        self.error: str | None = None

    def run(self) -> None:
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                output = run_adb(self.adb, self.serial, "exec-out", "dumpsys", "sensorservice")
                sensors = parse_metadata(output)
                events = []
                for event in parse_events(output, sensors):
                    key = (event["sensorId"], event["timestamp"], repr(event["rawValues"]))
                    if key in self.seen:
                        continue
                    self.seen.add(key)
                    self.seen_order.append(key)
                    events.append(event)
                while len(self.seen_order) > 20000:
                    self.seen.discard(self.seen_order.popleft())
                self.hub.update(sensors, events)
                self.error = None
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                self.error = str(exc)
            self.stop_event.wait(max(0.01, self.interval - (time.monotonic() - started)))

    def stop(self) -> None:
        self.stop_event.set()


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADB motion sensors</title>
<style>
:root{color-scheme:dark;font:14px system-ui,sans-serif}body{margin:0;background:#101318;color:#e8edf2}
header{position:sticky;top:0;background:#171b22;border-bottom:1px solid #303743;padding:16px 20px;z-index:2}
h1{font-size:20px;margin:0 0 5px}#status{color:#9aa7b4;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px;padding:16px}
article{background:#191e26;border:1px solid #303743;border-radius:8px;padding:13px;overflow:hidden}
h2{font-size:15px;margin:0 0 3px;color:#8bd5ff;overflow-wrap:anywhere}
.meta{color:#9aa7b4;font-size:11px;line-height:1.45;margin-bottom:9px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td{padding:3px 0;border-bottom:1px solid #282e38}td:first-child{color:#9aa7b4;width:42%}
td:last-child{text-align:right;color:#d8f5c0;overflow-wrap:anywhere}.empty{padding:30px;color:#9aa7b4}
.tabbar{display:flex;gap:8px;padding:12px 16px 0;background:#101318}.tab{border:1px solid #303743;background:#191e26;color:#9aa7b4;padding:8px 14px;border-radius:7px 7px 0 0;cursor:pointer}.tab.active{background:#26384a;color:#e8edf2;border-bottom-color:#26384a}.panel{min-height:calc(100vh - 130px)}.hidden{display:none}
.motion-toolbar{display:flex;align-items:center;gap:12px;padding:16px 16px 0;color:#9aa7b4;font-size:12px}.motion-toolbar button{border:1px solid #3d566e;background:#1c2a37;color:#d8f5c0;border-radius:6px;padding:7px 11px;cursor:pointer}.motion-toolbar span{overflow-wrap:anywhere}.scene-wrap{padding:16px}.scene{display:block;width:100%;height:min(70vh,720px);background:#0b0e13;border:1px solid #303743;border-radius:8px}
</style></head><body><header><h1>ADB motion sensors</h1><div id="status">Connecting…</div></header>
<nav class="tabbar"><button class="tab active" data-tab="rawTab">Raw sensors</button><button class="tab" data-tab="motionTab">3D motion</button></nav>
<main class="panel" id="rawTab"><div class="grid" id="grid"><div class="empty">Loading sensor inventory…</div></div></main>
<main class="panel hidden" id="motionTab"><div class="motion-toolbar"><button id="resetMotion" type="button">Reset path</button><span id="motionStatus">Waiting for acceleration events…</span></div><div class="scene-wrap"><canvas class="scene" id="scene"></canvas></div></main>
<script>
const meta=new Map(),latest=new Map(),grid=document.querySelector('#grid'),status=document.querySelector('#status');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function card(m){const x=latest.get(m.sensorId)||{},v=x.values||[];
return '<article><h2>'+esc(m.name)+'</h2><div class="meta">'+esc(m.typeName)+' · '+esc(m.vendor)+' · '+esc(m.sensorId)+
'<br>'+esc(m.descriptor)+'</div><table><tr><td>timestamp</td><td>'+esc(x.timestamp||'—')+
'</td></tr><tr><td>wall time</td><td>'+esc(x.wallTime||'—')+'</td></tr>'+
(v.length?v.map((n,i)=>'<tr><td>values['+i+']</td><td>'+esc(n)+'</td></tr>').join(''):
'<tr><td colspan="2">no exposed recent event</td></tr>')+'</table></article>'}
let renderPending=false;
function render(){renderPending=false;grid.innerHTML=[...meta.values()].map(card).join('')||'<div class="empty">No motion sensors found.</div>'}
function scheduleRender(){if(!renderPending){renderPending=true;requestAnimationFrame(render)}}
const canvas=document.querySelector('#scene'),ctx=canvas.getContext('2d'),motionStatus=document.querySelector('#motionStatus');
const MAX_DT_SECONDS=0.25;
const motion={path:[],lastT:null,v:[0,0,0],p:[0,0,0],gravity:null,quietFor:0,sensorId:null,source:'waiting'};
function resetMotion(){motion.path=[];motion.lastT=null;motion.v=[0,0,0];motion.p=[0,0,0];motion.gravity=null;motion.quietFor=0;motion.sensorId=null;motion.source='waiting';motionStatus.textContent='Waiting for acceleration events…'}
document.querySelector('#resetMotion').onclick=resetMotion;
function motionKind(x){
  if(x.type===10||/linear[_ ]acceleration/i.test((x.typeName||'')+' '+(x.name||'')))return 'linear';
  if(x.type===1)return 'accelerometer';
  return null;
}
function useMotionSource(kind,sensorId){
  if(motion.source===kind&&motion.sensorId===sensorId)return;
  resetMotion();motion.source=kind;motion.sensorId=sensorId;
}
function chooseMotionSource(events){
  let linear=null,accel=null;
  for(const x of events||[]){const kind=motionKind(x),t=Number(x.timestamp);if(!kind||!Number.isFinite(t)||!Array.isArray(x.values)||x.values.length<3)continue;const item={x,t};if(kind==='linear'&&(!linear||t>linear.t))linear=item;if(kind==='accelerometer'&&(!accel||t>accel.t))accel=item;}
  // dumpsys exposes linear acceleration only intermittently on some devices.
  // Prefer the continuously updating accelerometer so a path never changes
  // source or resets halfway through a movement.
  const chosen=accel?{kind:'accelerometer',...accel}:linear&&{kind:'linear',...linear};
  if(chosen)useMotionSource(chosen.kind,chosen.x.sensorId);
}
function motionEvent(x){
  if(!Array.isArray(x.values)||x.values.length<3)return;
  const kind=motionKind(x);if(!kind)return;
  const t=Number(x.timestamp),sample=x.values.slice(0,3).map(Number);if(!Number.isFinite(t)||sample.some(n=>!Number.isFinite(n)))return;
  if(motion.source==='waiting'){
    if(kind==='linear'&&[...meta.values()].some(sensor=>sensor.type===1))return;
    useMotionSource(kind,x.sensorId);
  }
  if(motion.source==='accelerometer'&&kind==='linear')return;
  if(motion.source==='linear'&&kind==='accelerometer')return;
  if(motion.sensorId!==x.sensorId)return;
  if(motion.lastT!==null&&t<=motion.lastT)return;
  const dt=motion.lastT===null?0:t-motion.lastT;motion.lastT=t;
  if(dt<=0)return;
  if(dt>MAX_DT_SECONDS){motion.v=[0,0,0];motion.gravity=null;motion.quietFor=0;return;}
  const a=sample.slice();
  if(motion.source==='accelerometer'){
    if(motion.gravity===null){motion.gravity=sample.slice();return;}
    const keep=Math.exp(-dt/0.8);
    for(let i=0;i<3;i++){motion.gravity[i]=keep*motion.gravity[i]+(1-keep)*sample[i];a[i]-=motion.gravity[i];}
  }
  for(let i=0;i<3;i++)a[i]=Math.max(-30,Math.min(30,Math.abs(a[i])<0.04?0:a[i]));
  motion.quietFor=Math.hypot(...a)<0.12?motion.quietFor+dt:0;
  const damping=Math.exp(-0.8*dt);
  for(let i=0;i<3;i++){const oldV=motion.v[i];motion.v[i]=(motion.v[i]+a[i]*dt)*damping;if(motion.quietFor>=0.35)motion.v[i]=0;motion.p[i]+=(oldV+motion.v[i])*0.5*dt}
  motion.path.push({t,x:motion.p[0],y:motion.p[1],z:motion.p[2]});
  while(motion.path.length>1&&t-motion.path[0].t>10)motion.path.shift();
  motionStatus.textContent=(motion.source==='linear'?'linear acceleration':'accelerometer (gravity removed)')+' · '+motion.path.length+' points · position ['+motion.p.map(n=>n.toFixed(2)).join(', ')+'] m';
}
function apply(x){if(x.sensors)x.sensors.forEach(m=>meta.set(m.sensorId,m));if(x.latest){x.latest.forEach(s=>latest.set(s.sensorId,s));chooseMotionSource(x.latest)}else if(x.sensorId){latest.set(x.sensorId,x);motionEvent(x)}scheduleRender()}
fetch('/api/sensors').then(r=>r.json()).then(x=>{apply(x);status.textContent=meta.size+' sensors · waiting for ADB events'}).catch(e=>status.textContent='Inventory error: '+e);
const stream=new EventSource('/stream');stream.onopen=()=>status.textContent=meta.size+' sensors · live';
stream.onerror=()=>status.textContent='ADB stream disconnected — retrying…';
stream.onmessage=e=>{apply(JSON.parse(e.data));status.textContent=meta.size+' sensors · live · '+new Date().toLocaleTimeString()};
document.querySelectorAll('.tab').forEach(button=>button.onclick=()=>{document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b===button));document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('hidden',p.id!==button.dataset.tab));resizeScene()});
function resizeScene(){const d=devicePixelRatio||1,r=canvas.getBoundingClientRect();canvas.width=Math.max(1,r.width*d);canvas.height=Math.max(1,r.height*d);ctx.setTransform(d,0,0,d,0,0)}
function project(p,yaw,pitch,scale,cx,cy){const cyaw=Math.cos(yaw),syaw=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);const x=p.x*cyaw-p.z*syaw,z=p.x*syaw+p.z*cyaw,y=p.y*cp-z*sp;return [cx+x*scale,cy-y*scale]}
function drawScene(){
  const w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h){requestAnimationFrame(drawScene);return}ctx.clearRect(0,0,w,h);ctx.fillStyle='#0b0e13';ctx.fillRect(0,0,w,h);
  const path=motion.path,extent=Math.max(2,...path.map(p=>Math.max(Math.abs(p.x),Math.abs(p.y),Math.abs(p.z))));const scale=Math.min(w,h)*0.34/extent,cx=w/2,cy=h/2+20,yaw=0.75,pitch=0.48;
  const line=(a,b,color,width=1)=>{const p=project(a,yaw,pitch,scale,cx,cy),q=project(b,yaw,pitch,scale,cx,cy);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke()};
  const size=extent*1.2;for(let i=-4;i<=4;i++){const d=i*size/4;line({x:-size,y:0,z:d},{x:size,y:0,z:d},'#202934');line({x:d,y:0,z:-size},{x:d,y:0,z:size},'#202934')}
  line({x:-size,y:0,z:0},{x:size,y:0,z:0},'#a34b52',2);line({x:0,y:-size,z:0},{x:0,y:size,z:0},'#4b9c66',2);line({x:0,y:0,z:-size},{x:0,y:0,z:size},'#477fa8',2);
  if(path.length){ctx.strokeStyle='#8bd5ff';ctx.lineWidth=3;ctx.lineJoin='round';ctx.beginPath();path.forEach((p,i)=>{const q=project(p,yaw,pitch,scale,cx,cy);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();const last=path[path.length-1],q=project(last,yaw,pitch,scale,cx,cy);ctx.fillStyle='#d8f5c0';ctx.beginPath();ctx.arc(q[0],q[1],5,0,Math.PI*2);ctx.fill();}
  ctx.fillStyle='#9aa7b4';ctx.font='12px system-ui';ctx.fillText('X red · Y green · Z blue · relative inertial path',16,24);if(!path.length)ctx.fillText('Move the phone to draw a path',16,h/2);
  requestAnimationFrame(drawScene);
}
window.addEventListener('resize',resizeScene);resizeScene();drawScene();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    hub: EventHub

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self.send_bytes("text/html; charset=utf-8", HTML.encode())
        elif path == "/api/sensors":
            self.send_bytes(
                "application/json; charset=utf-8",
                json.dumps(self.hub.snapshot(), separators=(",", ":")).encode(),
            )
        elif path == "/stream":
            self.stream()
        elif path == "/health":
            self.send_bytes("application/json; charset=utf-8", b'{"ok":true}')
        else:
            self.send_error(404)

    def send_bytes(self, content_type: str, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def stream(self) -> None:
        client = self.hub.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                try:
                    message = client.get(timeout=15)
                except queue.Empty:
                    message = ": keep-alive\n\n"
                self.wfile.write(message.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.hub.unsubscribe(client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", help="path to adb")
    parser.add_argument("--serial", help="ADB serial; defaults to the first authorized device")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=0.1, help="dumpsys polling interval in seconds")
    args = parser.parse_args()

    adb = find_adb(args.adb)
    serial = choose_device(adb, args.serial)
    hub = EventHub()
    poller = Poller(adb, serial, hub, args.interval)
    Handler.hub = hub
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    poller.start()
    print("ADB motion dashboard")
    print("  device: " + serial)
    print("  adb:    " + adb)
    print("  open:   http://%s:%d" % (args.host, args.port))
    print("  raw:    http://%s:%d/api/sensors" % (args.host, args.port))
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        poller.stop()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
