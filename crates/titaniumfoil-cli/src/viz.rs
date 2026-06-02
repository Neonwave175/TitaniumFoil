use std::f64::consts::PI;
use titaniumfoil_core::types::XfoilState;
use titaniumfoil_core::geometry::{load_naca, load_naca_n};
use titaniumfoil_core::panel::{calc_normals, calc_panel_angles, specal, ggcalc_finish};
use titaniumfoil_core::viscous::skin_friction_drag;
use titaniumfoil_metal::{MetalContext, panel_gpu::compute_panel_matrix_gpu};

fn build_state(naca: &str, nside: usize, re: f64) -> XfoilState {
    let mut state = XfoilState::default();
    if nside == 65 { load_naca(&mut state, naca); }
    else           { load_naca_n(&mut state, naca, nside); }
    state.op.qinf  = 1.0;
    state.op.reinf = re;
    state.op.minf  = 0.0;
    calc_normals(&mut state);
    calc_panel_angles(&mut state);
    state
}

fn solve(state: &mut XfoilState, ctx: &MetalContext, alpha_deg: f64) {
    state.op.alfa = alpha_deg * PI / 180.0;
    compute_panel_matrix_gpu(ctx, state);
    ggcalc_finish(state);
    specal(state);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: titaniumfoil-viz <naca> [alpha] [re] [nside]");
        eprintln!("  e.g. titaniumfoil-viz 4412 4.0 200000");
        std::process::exit(1);
    }

    let naca  = &args[1];
    let alpha: f64   = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let re:    f64   = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200_000.0);
    let nside: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(65);

    eprint!("  analysing NACA {} at α={:.1}° Re={:.0} … ", naca, alpha, re);

    let ctx = MetalContext::new();
    let mut state = build_state(naca, nside, re);
    solve(&mut state, &ctx, alpha);        // warm-up
    state = build_state(naca, nside, re);
    solve(&mut state, &ctx, alpha);        // real solve

    let n   = state.geom.n;
    let cdp = state.op.cdp;
    let cdf = skin_friction_drag(&state);
    let cd  = cdp + cdf;
    let cl  = state.op.cl;
    let cm  = state.op.cm;
    let ld  = if cd > 1e-9 { cl / cd } else { 0.0 };
    eprintln!("done  CL={:.4}  L/D={:.1}", cl, ld);

    let x:  Vec<f64> = (0..n).map(|i| state.geom.x[i]).collect();
    let y:  Vec<f64> = (0..n).map(|i| state.geom.y[i]).collect();
    let cp: Vec<f64> = (0..n).map(|i| state.vel.cpi[i]).collect();

    let html = generate_html(naca, alpha, re, &x, &y, &cp, cl, cd, cdp, cdf, cm, ld);

    let fname = format!("titaniumfoil-{}.html", naca);
    std::fs::write(&fname, &html).expect("failed to write HTML file");
    println!("  → {fname}");

    std::process::Command::new("open").arg(&fname).spawn().ok();
}

// ── HTML generator ────────────────────────────────────────────────────────────

fn js_array(v: &[f64]) -> String {
    let inner: Vec<String> = v.iter().map(|x| format!("{:.6}", x)).collect();
    format!("[{}]", inner.join(","))
}

fn generate_html(
    naca:  &str,
    alpha: f64, re: f64,
    x: &[f64], y: &[f64], cp: &[f64],
    cl: f64, cd: f64, cdp: f64, cdf: f64, cm: f64, ld: f64,
) -> String {
    let x_js  = js_array(x);
    let y_js  = js_array(y);
    let cp_js = js_array(cp);

    format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TitaniumFoil — NACA {naca}</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#0d1117;color:#e6edf3;font-family:'SF Mono','JetBrains Mono','Fira Code',monospace;padding:24px 32px}}
  h1{{font-size:1.25rem;font-weight:600;letter-spacing:.05em;color:#79c0ff;margin-bottom:4px}}
  .sub{{font-size:.75rem;color:#8b949e;margin-bottom:20px}}
  .stats{{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:20px}}
  .stat{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 16px}}
  .stat-label{{font-size:.65rem;color:#8b949e;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}}
  .stat-value{{font-size:1.1rem;font-weight:700;color:#e6edf3}}
  .stat-value.good{{color:#3fb950}}
  .canvases{{display:flex;flex-direction:column;gap:16px}}
  canvas{{border-radius:10px;display:block;width:100%;max-width:1000px}}
  .legend{{display:flex;gap:20px;margin-top:8px;font-size:.72rem;color:#8b949e}}
  .dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px;vertical-align:middle}}
</style>
</head>
<body>
<h1>NACA {naca}</h1>
<div class="sub">TitaniumFoil · GPU-accelerated XFOIL port for Apple Silicon</div>
<div class="stats">
  <div class="stat"><div class="stat-label">α (AoA)</div><div class="stat-value">{alpha:.1}°</div></div>
  <div class="stat"><div class="stat-label">Reynolds</div><div class="stat-value">{re:.0}</div></div>
  <div class="stat"><div class="stat-label">C_L</div><div class="stat-value {cl_cls}">{cl:.4}</div></div>
  <div class="stat"><div class="stat-label">C_D (total)</div><div class="stat-value">{cd:.5}</div></div>
  <div class="stat"><div class="stat-label">C_Dp (pressure)</div><div class="stat-value">{cdp:.5}</div></div>
  <div class="stat"><div class="stat-label">C_Df (friction)</div><div class="stat-value">{cdf:.5}</div></div>
  <div class="stat"><div class="stat-label">C_M</div><div class="stat-value">{cm:.4}</div></div>
  <div class="stat"><div class="stat-label">L/D</div><div class="stat-value good">{ld:.1}</div></div>
</div>
<div class="canvases">
  <canvas id="cvAirfoil" height="380"></canvas>
  <canvas id="cvCp" height="320"></canvas>
</div>
<div class="legend">
  <span><span class="dot" style="background:#79c0ff"></span>upper surface</span>
  <span><span class="dot" style="background:#56d364"></span>lower surface</span>
  <span style="margin-left:16px;color:#6e7681">pressure bars: outward = suction (blue) · inward = pressure (red)</span>
</div>
<script>
(function(){{
const X  = {x_js};
const Y  = {y_js};
const CP = {cp_js};
const N  = X.length;

// Find leading-edge index (minimum x)
let iLE = 0;
for (let i = 1; i < N; i++) if (X[i] < X[iLE]) iLE = i;

// Upper surface: LE index → 0 (reversed, so we go LE→TE over the top)
const upper = [];
for (let i = iLE; i >= 0; i--) upper.push({{x:X[i],y:Y[i],cp:CP[i]}});
// Lower surface: LE index → N-1
const lower = [];
for (let i = iLE; i < N; i++) lower.push({{x:X[i],y:Y[i],cp:CP[i]}});

// Pressure colormap: negative Cp = suction (blue), positive = pressure (red)
function cpColor(cp, alpha) {{
  alpha = alpha === undefined ? 1 : alpha;
  if (cp < 0) {{
    const t = Math.min(1, -cp / 2.8);
    const r = Math.round(255*(1-t*0.85));
    const g = Math.round(255*(1-t*0.55));
    return `rgba(${{r}},${{g}},255,${{alpha}})`;
  }} else {{
    const t = Math.min(1, cp / 1.0);
    const g = Math.round(255*(1-t*0.85));
    const b = Math.round(255*(1-t*0.95));
    return `rgba(255,${{g}},${{b}},${{alpha}})`;
  }}
}}

// ── Airfoil canvas ────────────────────────────────────────────────────────────
(function drawAirfoil() {{
  const canvas = document.getElementById('cvAirfoil');
  canvas.width = canvas.parentElement.clientWidth || 1000;
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  // Coordinate mapping: airfoil x∈[0,1], y∈[~-0.15,~0.15]
  const padL=80, padR=40, padT=60, padB=60;
  const xMin=0, xMax=1;
  const yRange = 0.40;  // half-range shown
  const scX = (W-padL-padR)/(xMax-xMin);
  const scY = (H-padT-padB)/(2*yRange);
  const mapX = x => padL + (x-xMin)*scX;
  const mapY = y => padT + (yRange - y)*scY;

  // ── Grid lines ──
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth = 1;
  [0,0.25,0.5,0.75,1.0].forEach(xg => {{
    ctx.beginPath();
    ctx.moveTo(mapX(xg), padT);
    ctx.lineTo(mapX(xg), H-padB);
    ctx.stroke();
  }});
  [-0.1,0,0.1].forEach(yg => {{
    ctx.beginPath();
    ctx.moveTo(padL, mapY(yg));
    ctx.lineTo(W-padR, mapY(yg));
    ctx.stroke();
  }});

  // Chord line
  ctx.strokeStyle = '#30363d';
  ctx.setLineDash([4,4]);
  ctx.beginPath();
  ctx.moveTo(mapX(0), mapY(0));
  ctx.lineTo(mapX(1), mapY(0));
  ctx.stroke();
  ctx.setLineDash([]);

  // ── Freestream arrow ──
  const ALPHA_RAD = {alpha} * Math.PI / 180;
  const arrowLen = 55;
  const ax0 = padL - 50, ay0 = mapY(0);
  const ax1 = ax0 + arrowLen*Math.cos(ALPHA_RAD);
  const ay1 = ay0 - arrowLen*Math.sin(ALPHA_RAD);
  ctx.strokeStyle = '#8b949e'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(ax0,ay0); ctx.lineTo(ax1,ay1); ctx.stroke();
  // arrowhead
  const aw=7,ah=3;
  const ang = Math.atan2(ay0-ay1, ax1-ax0);
  ctx.beginPath();
  ctx.moveTo(ax1,ay1);
  ctx.lineTo(ax1-aw*Math.cos(ang-ah), ay1+aw*Math.sin(ang-ah));
  ctx.lineTo(ax1-aw*Math.cos(ang+ah), ay1+aw*Math.sin(ang+ah));
  ctx.closePath(); ctx.fillStyle='#8b949e'; ctx.fill();
  // V∞ label
  ctx.fillStyle='#8b949e'; ctx.font='11px monospace';
  ctx.fillText('V∞', ax0-28, ay0+4);

  // ── Pressure comb ──
  function drawComb(surface, flip) {{
    for (let i=0; i<surface.length; i++) {{
      const pt = surface[i];
      const prev = surface[Math.max(0,i-1)];
      const next = surface[Math.min(surface.length-1,i+1)];
      // Tangent
      const tx = next.x - prev.x, ty = next.y - prev.y;
      const tl = Math.sqrt(tx*tx+ty*ty)||1;
      // Outward normal (left-hand turn of tangent = outward for CCW contour)
      let nx = -ty/tl, ny = tx/tl;
      if (flip) {{ nx=-nx; ny=-ny; }}
      // Bar length proportional to |Cp|, direction sign encodes sign of Cp
      const scale = 0.08;
      const cpSign = (pt.cp < 0) ? 1 : -1;  // suction bars point outward
      const barLen = Math.abs(pt.cp) * scale * cpSign;
      const x0 = mapX(pt.x), y0 = mapY(pt.y);
      const x1 = mapX(pt.x + nx*barLen), y1 = mapY(pt.y + ny*barLen);
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.strokeStyle = cpColor(pt.cp, 0.85);
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }}
  }}
  drawComb(upper, false);
  drawComb(lower, true);

  // ── Airfoil fill then outline ──
  ctx.beginPath();
  ctx.moveTo(mapX(X[0]), mapY(Y[0]));
  for (let i=1;i<N;i++) ctx.lineTo(mapX(X[i]), mapY(Y[i]));
  ctx.closePath();
  // Subtle gradient fill
  const grad = ctx.createLinearGradient(mapX(0),mapY(0.05),mapX(0),mapY(-0.05));
  grad.addColorStop(0,'#1c2128');
  grad.addColorStop(1,'#161b22');
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.strokeStyle='#c9d1d9'; ctx.lineWidth=1.8; ctx.stroke();

  // ── Stagnation point dot ──
  // Rough stagnation: maximum Cp near the LE
  let stag = 0; let maxCp=-99;
  for (let i=0;i<N;i++) {{
    if (X[i]<0.15 && CP[i]>maxCp) {{ maxCp=CP[i]; stag=i; }}
  }}
  ctx.beginPath();
  ctx.arc(mapX(X[stag]), mapY(Y[stag]), 3.5, 0, 2*Math.PI);
  ctx.fillStyle='#f78166'; ctx.fill();

  // ── Colorbar ──
  const barX=W-28, barY=padT, barH=H-padT-padB, barW=12;
  const cbGrad = ctx.createLinearGradient(0,barY,0,barY+barH);
  cbGrad.addColorStop(0.0, 'rgba(255,60,60,1)');   // high Cp (pressure)
  cbGrad.addColorStop(0.4, 'rgba(255,255,255,0.15)'); // Cp≈0
  cbGrad.addColorStop(1.0, 'rgba(60,130,255,1)');  // low Cp (suction)
  ctx.fillStyle=cbGrad;
  ctx.beginPath();
  ctx.roundRect(barX,barY,barW,barH,4);
  ctx.fill();
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1; ctx.stroke();
  ctx.fillStyle='#8b949e'; ctx.font='10px monospace';
  ctx.fillText('Cp+', barX-2, barY-6);
  ctx.fillText('Cp−', barX-2, barY+barH+14);

  // ── Axis labels ──
  ctx.fillStyle='#8b949e'; ctx.font='11px monospace';
  [0,0.25,0.5,0.75,1.0].forEach(xg => {{
    ctx.fillText(xg.toFixed(2), mapX(xg)-12, H-padB+18);
  }});
  ctx.fillText('x/c', W/2-12, H-padB+34);

  // Title
  ctx.fillStyle='#e6edf3'; ctx.font='bold 13px monospace';
  ctx.fillText('Pressure distribution — NACA {naca}  α={alpha:.1}°  Re={re:.0}', padL, 20);
  ctx.fillStyle='#8b949e'; ctx.font='11px monospace';
  ctx.fillText('bars: outward=suction · inward=pressure', padL, 36);
}})();

// ── Cp distribution canvas ────────────────────────────────────────────────────
(function drawCpChart() {{
  const canvas = document.getElementById('cvCp');
  canvas.width = canvas.parentElement.clientWidth || 1000;
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0,0,W,H);

  const padL=60,padR=30,padT=50,padB=50;
  const chartW = W-padL-padR, chartH = H-padT-padB;

  // -Cp range
  const allNegCp = [...upper,...lower].map(p=>-p.cp);
  let cpMin = Math.min(...allNegCp);
  let cpMax = Math.max(...allNegCp);
  // nice padding
  cpMin = Math.floor((cpMin-0.2)*4)/4;
  cpMax = Math.ceil ((cpMax+0.2)*4)/4;

  const mapX = x => padL + x*chartW;
  const mapY = v => padT + (cpMax-v)/(cpMax-cpMin)*chartH;

  // ── Grid ──
  ctx.strokeStyle='#21262d'; ctx.lineWidth=1;
  const nYticks = 6;
  for (let i=0;i<=nYticks;i++) {{
    const v = cpMin + (cpMax-cpMin)*i/nYticks;
    const y = mapY(v);
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(W-padR,y); ctx.stroke();
    ctx.fillStyle='#6e7681'; ctx.font='10px monospace';
    ctx.fillText(v.toFixed(2), 2, y+4);
  }}
  [0,0.25,0.5,0.75,1.0].forEach(xg => {{
    ctx.beginPath(); ctx.moveTo(mapX(xg),padT); ctx.lineTo(mapX(xg),H-padB); ctx.stroke();
    ctx.fillStyle='#6e7681'; ctx.font='10px monospace';
    ctx.fillText(xg.toFixed(2), mapX(xg)-12, H-padB+16);
  }});

  // Cp=0 reference line
  if (0>=cpMin && 0<=cpMax) {{
    ctx.strokeStyle='#3d4450'; ctx.lineWidth=1.5; ctx.setLineDash([5,3]);
    ctx.beginPath(); ctx.moveTo(padL,mapY(0)); ctx.lineTo(W-padR,mapY(0)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#6e7681'; ctx.font='10px monospace';
    ctx.fillText('0', 2, mapY(0)+4);
  }}

  // ── Upper surface (blue) ──
  ctx.beginPath();
  ctx.strokeStyle='#79c0ff'; ctx.lineWidth=2.2;
  upper.forEach((pt,i)=>{{
    if(i===0) ctx.moveTo(mapX(pt.x),mapY(-pt.cp));
    else      ctx.lineTo(mapX(pt.x),mapY(-pt.cp));
  }});
  ctx.stroke();

  // Fill under upper curve (suction region above Cp=0 line)
  ctx.beginPath();
  upper.forEach((pt,i)=>{{
    if(i===0) ctx.moveTo(mapX(pt.x),mapY(-pt.cp));
    else      ctx.lineTo(mapX(pt.x),mapY(-pt.cp));
  }});
  ctx.lineTo(mapX(upper[upper.length-1].x),mapY(0));
  ctx.lineTo(mapX(upper[0].x),mapY(0));
  ctx.closePath();
  ctx.fillStyle='rgba(121,192,255,0.07)'; ctx.fill();

  // ── Lower surface (green) ──
  ctx.beginPath();
  ctx.strokeStyle='#56d364'; ctx.lineWidth=2.2;
  lower.forEach((pt,i)=>{{
    if(i===0) ctx.moveTo(mapX(pt.x),mapY(-pt.cp));
    else      ctx.lineTo(mapX(pt.x),mapY(-pt.cp));
  }});
  ctx.stroke();

  // Fill under lower curve
  ctx.beginPath();
  lower.forEach((pt,i)=>{{
    if(i===0) ctx.moveTo(mapX(pt.x),mapY(-pt.cp));
    else      ctx.lineTo(mapX(pt.x),mapY(-pt.cp));
  }});
  ctx.lineTo(mapX(lower[lower.length-1].x),mapY(0));
  ctx.lineTo(mapX(lower[0].x),mapY(0));
  ctx.closePath();
  ctx.fillStyle='rgba(86,211,100,0.06)'; ctx.fill();

  // ── Stagnation dot on chart ──
  const stagPt = lower[0];  // LE ≈ start of lower surface
  ctx.beginPath();
  ctx.arc(mapX(stagPt.x),mapY(-stagPt.cp),4,0,2*Math.PI);
  ctx.fillStyle='#f78166'; ctx.fill();

  // ── Axes ──
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(padL,padT); ctx.lineTo(padL,H-padB); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(padL,H-padB); ctx.lineTo(W-padR,H-padB); ctx.stroke();

  // ── Labels ──
  ctx.fillStyle='#8b949e'; ctx.font='11px monospace';
  ctx.fillText('x/c', W/2-10, H-padB+34);
  ctx.save(); ctx.translate(14,H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('−Cp', -16, 0); ctx.restore();

  // Title
  ctx.fillStyle='#e6edf3'; ctx.font='bold 13px monospace';
  ctx.fillText('−Cp distribution  NACA {naca}  α={alpha:.1}°  Re={re:.0}', padL, 20);

  // Legend
  ctx.fillStyle='#79c0ff'; ctx.fillRect(padL, padT+8, 20, 2);
  ctx.fillStyle='#8b949e'; ctx.font='11px monospace'; ctx.fillText('upper', padL+24, padT+13);
  ctx.fillStyle='#56d364'; ctx.fillRect(padL+80, padT+8, 20, 2);
  ctx.fillStyle='#8b949e'; ctx.fillText('lower', padL+104, padT+13);
}})();

}})();
</script>
</body>
</html>"#,
        naca  = naca,
        alpha = alpha,
        re    = re,
        cl    = cl, cd = cd, cdp = cdp, cdf = cdf, cm = cm, ld = ld,
        cl_cls = if cl > 0.0 { "good" } else { "" },
        x_js  = x_js,
        y_js  = y_js,
        cp_js = cp_js,
    )
}
