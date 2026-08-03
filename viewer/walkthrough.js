/**
 * Walkthrough viewer logic (PLAN.md §1.1, §3.8).
 *
 * Consumes the scene manifest written by `recon.io.viewer`. Everything about
 * where rooms sit comes from that file; nothing about the house is hardcoded.
 */

const HUD = document.getElementById("hud");
const ERR = document.getElementById("err");
const ROOMS = document.getElementById("rooms");

/** Babylon is left-handed by default; our scenes are right-handed, Y-up. */
export function applyFrame(scene, frame) {
  const conversion = frame && frame.viewer_conversion;
  if (conversion === "negate_z") {
    // useRightHandedSystem is the supported way to do this — negating Z on
    // every mesh would also flip winding and turn the room inside out.
    scene.useRightHandedSystem = true;
    return "right-handed";
  }
  if (conversion == null) {
    throw new Error(
      "scene manifest does not state its coordinate frame; refusing to guess. " +
      "A wrong guess renders a mirrored house that looks correct."
    );
  }
  throw new Error(`unsupported viewer_conversion: ${conversion}`);
}

/** Room -> house transform, matching recon.schemas.SE2Transform. */
export function roomMatrix(BABYLON, room) {
  const c = Math.cos(room.yaw_rad), s = Math.sin(room.yaw_rad);
  const k = room.scale ?? 1;
  return BABYLON.Matrix.FromValues(
    c * k, 0, s * k, 0,
    0, k, 0, 0,
    -s * k, 0, c * k, 0,
    room.tx, 0, room.tz, 1,
  );
}

export function roomToHouse(room, x, z) {
  const c = Math.cos(room.yaw_rad), s = Math.sin(room.yaw_rad);
  const k = room.scale ?? 1;
  return [k * (c * x + s * z) + room.tx, k * (-s * x + c * z) + room.tz];
}

/** Floor-plan overlay: every room's footprint in the shared house frame. */
export function planSvg(scene, currentRoomId) {
  const polys = scene.rooms.map((r) => ({
    id: r.room_id,
    pts: r.footprint.map(([x, z]) => roomToHouse(r, x, z)),
  }));
  const all = polys.flatMap((p) => p.pts);
  if (!all.length) return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"></svg>';
  const xs = all.map((p) => p[0]), zs = all.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minZ = Math.min(...zs), maxZ = Math.max(...zs);
  const pad = 0.4;
  const w = maxX - minX + 2 * pad, h = maxZ - minZ + 2 * pad;
  const body = polys.map((p) => {
    const d = p.pts.map(([x, z]) => `${(x - minX + pad).toFixed(2)},${(maxZ - z + pad).toFixed(2)}`).join(" ");
    const here = p.id === currentRoomId;
    return `<polygon points="${d}" fill="${here ? "#2dd4bf40" : "#ffffff14"}" ` +
           `stroke="${here ? "#2dd4bf" : "#ffffff59"}" stroke-width="0.06"/>`;
  }).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w.toFixed(2)} ${h.toFixed(2)}">${body}</svg>`;
}

function fail(message) {
  ERR.style.display = "block";
  ERR.textContent = message;
  HUD.textContent = "failed to load";
  throw new Error(message);
}

async function main() {
  const BABYLON = window.BABYLON;
  if (!BABYLON) fail("Babylon.js was not loaded — check viewer/vendor/babylon.js");

  const params = new URLSearchParams(location.search);
  const manifestUrl = params.get("scene") || "./scene.json";
  let scene_;
  try {
    scene_ = await (await fetch(manifestUrl)).json();
  } catch (e) {
    fail(`could not load scene manifest ${manifestUrl}: ${e.message}`);
  }

  const canvas = document.getElementById("view");
  const engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true });
  const scene = new BABYLON.Scene(engine);
  applyFrame(scene, scene_.frame);
  scene.clearColor = new BABYLON.Color4(0.04, 0.05, 0.07, 1);

  const camera = new BABYLON.UniversalCamera("cam", new BABYLON.Vector3(0, 1.6, 0), scene);
  camera.attachControl(canvas, true);
  camera.speed = 0.12;
  camera.minZ = 0.05;
  camera.inertia = 0.82;
  // Walkthrough, not orbit: the deliverable is "what does my house look like",
  // and an orbit camera outside the building answers a different question.
  camera.keysUp = [87, 38]; camera.keysDown = [83, 40];
  camera.keysLeft = [65, 37]; camera.keysRight = [68, 39];

  new BABYLON.HemisphericLight("sky", new BABYLON.Vector3(0, 1, 0), scene).intensity = 0.9;

  let loaded = 0;
  for (const room of scene_.rooms) {
    const splat = room.artifacts.spz || room.artifacts.ply;
    if (!splat) continue;
    try {
      const result = await BABYLON.SceneLoader.ImportMeshAsync("", "", splat, scene);
      const node = new BABYLON.TransformNode(room.room_id, scene);
      node.setPreTransformMatrix
        ? node.setPreTransformMatrix(roomMatrix(BABYLON, room))
        : (node.position = new BABYLON.Vector3(room.tx, 0, room.tz),
           node.rotation = new BABYLON.Vector3(0, room.yaw_rad, 0),
           node.scaling = new BABYLON.Vector3(room.scale, room.scale, room.scale));
      for (const mesh of result.meshes) mesh.parent = node;
      loaded += 1;
    } catch (e) {
      console.warn(`room ${room.room_id}: ${e.message}`);
    }
  }

  const goTo = (room) => {
    const [x, z] = roomToHouse(room, room.spawn[0], room.spawn[2]);
    camera.position = new BABYLON.Vector3(x, room.spawn[1], z);
    document.getElementById("planSvg").outerHTML =
      planSvg(scene_, room.room_id).replace("<svg", '<svg id="planSvg"');
    for (const b of ROOMS.querySelectorAll("button"))
      b.setAttribute("aria-current", String(b.dataset.room === room.room_id));
  };

  for (const room of scene_.rooms) {
    const b = document.createElement("button");
    b.textContent = `${room.room_id} · ${room.area_m2.toFixed(1)} m²`;
    b.dataset.room = room.room_id;
    b.onclick = () => goTo(room);
    ROOMS.appendChild(b);
  }
  ROOMS.hidden = scene_.rooms.length < 2;
  document.getElementById("plan").hidden = false;

  HUD.innerHTML =
    `<b>${scene_.rooms.length}</b> room${scene_.rooms.length === 1 ? "" : "s"} · ` +
    `<b>${scene_.total_area_m2.toFixed(1)} m²</b><br>` +
    (loaded ? "" : "no splat artifacts in this scene — showing the plan only<br>") +
    `<span style="opacity:.7">WASD / arrows to move, drag to look</span>`;

  if (scene_.rooms.length) goTo(scene_.rooms[0]);
  engine.runRenderLoop(() => scene.render());
  addEventListener("resize", () => engine.resize());
  canvas.focus();
}

if (typeof document !== "undefined" && document.getElementById("view")) main();
