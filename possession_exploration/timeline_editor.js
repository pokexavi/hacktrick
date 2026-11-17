// The 2D replay of the app, with a possession bar whose boundaries you can drag.
// It sends the edited runs ([[start, stop, label]]) back to Streamlit after every change.

const COLORS = {H: "#3f8fe0", A: "#d94a4a", D: "#8a93a0", X: "#253246"};
const NAMES = {H: "H", A: "A", D: "dead ball", X: "no label"};
const GRAB_PX = 9;              // click within 9 px of a handle to grab it

export default function (component) {
  const {data, setStateValue, parentElement} = component;
  const D = data;
  const memory = (window.__possessionEditor = window.__possessionEditor || {});
  const saved = memory[D.key];
  let runs = saved ? saved.runs.map(r => r.slice()) : D.runs.map(r => r.slice());
  let frame = saved ? saved.frame : Math.max(0, Math.min(D.t.length - 1, D.initial || 0));
  let selected = saved ? saved.selected : null;
  let drag = null, timer = null, pending = null;
  const n = D.t.length;

  const old = parentElement.querySelector(".pe-root");
  if (old) old.remove();
  const root = document.createElement("div");
  root.className = "pe-root";
  root.innerHTML = `
    <div class="pe-controls"><button class="pe-play">Play</button><span class="pe-clock"></span>
      <span class="pe-now"></span><span class="pe-event"></span></div>
    <canvas class="pe-pitch" width="1050" height="680"></canvas>
    <canvas class="pe-bar" width="1050" height="84" tabindex="0"></canvas>
    <div class="pe-help">Drag a white handle to move a boundary · drop it on the next one to delete a
      segment · click a handle, then ← / → for one frame (Shift: 5) · click the bar to seek · Space: play</div>
    <div class="pe-legend"><b style="color:${COLORS.H}">●</b> H: ${D.teams[0]} &nbsp;
      <b style="color:${COLORS.A}">●</b> A: ${D.teams[1]} &nbsp; <b style="color:${COLORS.D}">●</b> dead ball
      &nbsp; <b style="color:#ff4fd8">▲</b> the transition under review &nbsp; <b style="color:#7fdbff">|</b> replay</div>`;
  parentElement.appendChild(root);
  const pitch = root.querySelector(".pe-pitch"), ctx = pitch.getContext("2d");
  const bar = root.querySelector(".pe-bar"), bx = bar.getContext("2d");
  const play = root.querySelector(".pe-play");

  function remember() { memory[D.key] = {runs, frame, selected}; }

  // Runs
  function labelAt(i) { for (const r of runs) if (i >= r[0] && i < r[1]) return r[2]; return "X"; }
  function normalize() {
    // Remove empty segments and join neighbours with the same label. Keep the handle selected if it still exists.
    const at = selected === null ? null : runs[selected][0];
    let out = runs.filter(r => r[1] > r[0]);
    const merged = [];
    for (const r of out) {
      const last = merged[merged.length - 1];
      if (last && last[2] === r[2]) last[1] = r[1]; else merged.push(r.slice());
    }
    runs = merged;
    selected = null;
    if (at !== null) for (let k = 1; k < runs.length; k++) if (runs[k][0] === at) selected = k;
  }
  function commit() {
    normalize();
    remember();
    setStateValue("runs", runs);
  }
  function moveBoundary(k, to) {
    const lo = runs[k - 1][0], hi = runs[k][1];
    to = Math.max(lo, Math.min(hi, Math.round(to)));
    runs[k - 1][1] = to;
    runs[k][0] = to;
    frame = Math.max(0, Math.min(n - 1, to));
  }

  // Drawing
  const W = bar.width, H = bar.height, TOP = 14, BOT = 66;
  const xOf = i => i / n * W;
  const toIndex = e => (e.offsetX * bar.width / bar.clientWidth) / W * n;
  function line(x1, y1, x2, y2) { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }
  function circle(x, y, r, fill) { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); if (fill) ctx.fill(); else ctx.stroke(); }

  function drawPitch() {
    const w = pitch.width, h = pitch.height, m = 24, px = x => m + x * (w - 2 * m), py = y => m + y * (h - 2 * m);
    ctx.fillStyle = "#17643e"; ctx.fillRect(0, 0, w, h); ctx.strokeStyle = "rgba(255,255,255,.8)"; ctx.lineWidth = 2;
    ctx.strokeRect(m, m, w - 2 * m, h - 2 * m); line(w / 2, m, w / 2, h - m); circle(w / 2, h / 2, 65, false);
    ctx.strokeRect(m, m + 145, 160, h - 2 * m - 290); ctx.strokeRect(w - m - 160, m + 145, 160, h - 2 * m - 290);
    ctx.strokeRect(m, m + 225, 55, h - 2 * m - 450); ctx.strokeRect(w - m - 55, m + 225, 55, h - 2 * m - 450);
    const players = D.xy[frame];
    for (let p = 0; p < players.length; p++) {
      const pos = players[p];
      if (pos[0] === null || pos[1] === null) continue;
      ctx.fillStyle = D.player_team[p] === 0 ? COLORS.H : COLORS.A; circle(px(pos[0]), py(pos[1]), 9, true);
      ctx.fillStyle = "#07131d"; ctx.font = "10px system-ui"; ctx.textAlign = "center";
      ctx.fillText(String(p + 1), px(pos[0]), py(pos[1]) + 3);
    }
    const ball = D.ball[frame];
    if (ball[0] !== null && ball[1] !== null) { ctx.fillStyle = "white"; circle(px(ball[0]), py(ball[1]), 5, true); ctx.strokeStyle = "#111"; ctx.stroke(); }
  }

  function drawBar() {
    bx.clearRect(0, 0, W, H);
    for (const r of runs) { bx.fillStyle = COLORS[r[2]] || COLORS.X; bx.fillRect(xOf(r[0]), TOP, Math.max(1, xOf(r[1]) - xOf(r[0])), BOT - TOP); }
    for (const e of D.events) {
      const i = Math.round((e.t - D.t[0]) * D.fps);
      if (i >= 0 && i < n) { bx.fillStyle = "#f5d87a"; bx.fillRect(xOf(i), TOP, 2, 10); }
    }
    for (let k = 1; k < runs.length; k++) {               // handles
      const x = xOf(runs[k][0]), active = k === selected || (drag && drag.k === k);
      bx.fillStyle = active ? "#ffd400" : "#ffffff";
      bx.fillRect(x - 1.5, TOP - 4, 3, BOT - TOP + 8);
      bx.fillRect(x - 5, TOP - 12, 10, 9);
    }
    if (D.focus !== null && D.focus !== undefined) {      // the transition under review
      const x = xOf(D.focus);
      bx.fillStyle = "#ff4fd8"; bx.beginPath(); bx.moveTo(x, BOT + 2); bx.lineTo(x - 7, H - 2); bx.lineTo(x + 7, H - 2); bx.fill();
    }
    bx.fillStyle = "#7fdbff"; bx.fillRect(xOf(frame) - 0.5, TOP, 1.5, BOT - TOP);     // playhead
  }

  function draw() {
    drawPitch(); drawBar();
    const t = D.t[frame];
    root.querySelector(".pe-clock").textContent =
      `P${D.period} ${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}.${String(Math.floor(t * 100) % 100).padStart(2, "0")}`;
    const where = selected !== null ? ` · handle ${labelAt(runs[selected][0] - 1)}→${labelAt(runs[selected][0])}` : "";
    root.querySelector(".pe-now").textContent = "Now: " + NAMES[labelAt(frame)] + where;
    const near = D.events.filter(e => Math.abs(e.t - t) < 0.6);
    root.querySelector(".pe-event").textContent = near.length ? "Event: " + near.map(e => e.type).join(", ") : "";
  }

  // Mouse and keyboard
  function handleNear(e) {
    const perPx = n / bar.clientWidth;                     // frames per pixel
    const i = toIndex(e);
    let best = null, dist = GRAB_PX * perPx;
    for (let k = 1; k < runs.length; k++) {
      const d = Math.abs(runs[k][0] - i);
      if (d <= dist) { best = k; dist = d; }
    }
    return best;
  }
  bar.addEventListener("pointerdown", e => {
    bar.focus();
    const k = handleNear(e);
    if (k !== null) { drag = {k}; selected = k; bar.setPointerCapture(e.pointerId); }
    else { selected = null; frame = Math.max(0, Math.min(n - 1, Math.floor(toIndex(e)))); remember(); }
    draw();
  });
  bar.addEventListener("pointermove", e => {
    if (drag) { moveBoundary(drag.k, toIndex(e)); draw(); }
    else bar.style.cursor = handleNear(e) !== null ? "ew-resize" : "pointer";
  });
  bar.addEventListener("pointerup", () => { if (drag) { drag = null; commit(); draw(); } });
  bar.addEventListener("keydown", e => {
    if (e.key === " ") { e.preventDefault(); toggle(); return; }
    if (selected === null || (e.key !== "ArrowLeft" && e.key !== "ArrowRight")) return;
    e.preventDefault();
    const step = (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 5 : 1);
    moveBoundary(selected, runs[selected][0] + step);
    draw();
    clearTimeout(pending);
    pending = setTimeout(() => { commit(); draw(); }, 350);    // wait a bit so holding the key doesn't rerun every press
  });
  function toggle() {
    if (timer) { clearInterval(timer); timer = null; play.textContent = "Play"; }
    else { timer = setInterval(() => { frame = (frame + 1) % n; draw(); }, 1000 / D.fps); play.textContent = "Pause"; }
  }
  play.addEventListener("click", toggle);
  draw();
  return () => { if (timer) clearInterval(timer); clearTimeout(pending); remember(); };
}
