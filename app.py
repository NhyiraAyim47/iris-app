import json
import struct
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
MODEL = json.loads((BASE / "iris_model.json").read_text())
TREE = MODEL["tree"]

app = FastAPI(title="Iris Species Classifier")


class Flower(BaseModel):
    sepal_length: float = Field(gt=0, le=20)
    sepal_width: float = Field(gt=0, le=20)
    petal_length: float = Field(gt=0, le=20)
    petal_width: float = Field(gt=0, le=20)


def as_float32(v: float) -> float:
    # scikit-learn compares float32 values, so do the same for identical results
    return struct.unpack("f", struct.pack("f", v))[0]


def run_tree(x):
    """Follow the trained decision tree's yes/no questions down to a leaf."""
    node, path = 0, []
    while TREE["left"][node] != -1:
        f = TREE["feature"][node]
        threshold = TREE["threshold"][node]
        go_left = as_float32(x[f]) <= threshold
        path.append(
            {
                "feature": MODEL["features"][f],
                "value": x[f],
                "threshold": round(threshold, 3),
                "left": go_left,
            }
        )
        node = TREE["left"][node] if go_left else TREE["right"][node]
    counts = TREE["value"][node]
    total = sum(counts)
    return [c / total for c in counts], path


@app.post("/predict")
def predict(flower: Flower):
    x = [flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width]
    probs, path = run_tree(x)
    best = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "prediction": MODEL["classes"][best],
        "probabilities": {c: round(p, 3) for c, p in zip(MODEL["classes"], probs)},
        "path": path,
    }


BOOT = {
    "classes": MODEL["classes"],
    "accuracy": MODEL["accuracy"],
    "train_size": MODEL["train_size"],
    "test_size": MODEL["test_size"],
    "points": MODEL["points"],
}

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Iris Species Classifier</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #F0EEF7;
    --panel: #FFFFFF;
    --ink: #1F1B3D;
    --muted: #5F5880;
    --line: #DDD8EC;
    --field: #F7F6FB;
    --indigo: #3B2D9B;
    --indigo-dark: #2B2078;
    --setosa: #1F9D8B;
    --versicolor: #6B4FD3;
    --virginica: #E0832B;
    --radius: 16px;
  }
  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body {
    margin: 0;
    background:
      radial-gradient(900px 420px at 100% -10%, rgba(107, 79, 211, 0.14), transparent 70%),
      radial-gradient(700px 360px at -10% 0%, rgba(31, 157, 139, 0.10), transparent 70%),
      var(--bg);
    color: var(--ink);
    font-family: "Figtree", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    font-size: 16px;
    line-height: 1.5;
  }
  .wrap { max-width: 1060px; margin: 0 auto; padding: 36px 20px 56px; }

  .top { display: flex; align-items: center; gap: 18px; margin-bottom: 28px; }
  .mark { flex: none; width: 68px; height: 68px; }
  h1 { font-size: clamp(27px, 4vw, 38px); line-height: 1.1; margin: 0 0 6px; font-weight: 800; letter-spacing: -0.02em; }
  .top p { margin: 0; color: var(--muted); max-width: 56ch; }

  .layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 22px; align-items: start; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); padding: 26px 28px; box-shadow: 0 1px 0 rgba(31, 27, 61, 0.03), 0 12px 28px -18px rgba(31, 27, 61, 0.25); }
  .card h2 { margin: 0 0 14px; font-size: 18px; font-weight: 700; }
  .wide { margin-top: 22px; }

  .examples { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 22px; }
  .examples .lbl { font-size: 14px; color: var(--muted); margin-right: 4px; }
  .chip { font: inherit; font-size: 14px; font-weight: 600; color: var(--ink); background: var(--field); border: 1.5px solid var(--line); border-radius: 999px; padding: 6px 14px 6px 10px; cursor: pointer; display: inline-flex; align-items: center; gap: 7px; transition: border-color 0.15s, background 0.15s; }
  .chip:hover { border-color: var(--indigo); background: #fff; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }
  .dot.setosa { background: var(--setosa); }
  .dot.versicolor { background: var(--versicolor); }
  .dot.virginica { background: var(--virginica); }

  .meas + .meas { margin-top: 20px; }
  .meas-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
  .meas-head label { font-weight: 600; font-size: 15px; }
  .num { position: relative; width: 112px; }
  .num input { width: 100%; font: inherit; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--ink); background: var(--field); border: 1.5px solid var(--line); border-radius: 10px; padding: 7px 38px 7px 12px; -moz-appearance: textfield; }
  .num input::-webkit-outer-spin-button, .num input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
  .num span { position: absolute; right: 12px; top: 50%; transform: translateY(-50%); font-size: 13px; color: var(--muted); pointer-events: none; }
  input[type="range"] { width: 100%; accent-color: var(--indigo); margin: 8px 0 0; }
  .ends { display: flex; justify-content: space-between; font-size: 12.5px; color: var(--muted); }
  :focus-visible { outline: 3px solid rgba(59, 45, 155, 0.4); outline-offset: 2px; }
  .num input:focus-visible { border-color: var(--indigo); outline-offset: 0; }

  .actions { display: flex; align-items: center; gap: 10px; margin-top: 26px; }
  button.primary { font: inherit; font-weight: 700; color: #fff; background: var(--indigo); border: 0; border-radius: 11px; padding: 13px 26px; cursor: pointer; transition: background 0.15s; }
  button.primary:hover { background: var(--indigo-dark); }
  button.primary:disabled { background: #9A90CF; cursor: progress; }
  button.ghost { font: inherit; font-weight: 600; color: var(--muted); background: none; border: 0; padding: 13px 8px; cursor: pointer; text-decoration: underline; text-underline-offset: 3px; }
  button.ghost:hover { color: var(--ink); }

  .result { position: sticky; top: 20px; min-height: 360px; }
  .empty { color: var(--muted); margin: 6px 0 0; }
  .verdict { display: flex; align-items: center; gap: 12px; }
  .verdict .dot { width: 16px; height: 16px; }
  .verdict strong { font-size: 28px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; }
  .about-species { margin: 6px 0 0; color: var(--muted); }
  .conf { margin: 14px 0 18px; font-size: 14.5px; color: var(--muted); }

  .bars { display: grid; gap: 11px; }
  .bar-row { display: grid; grid-template-columns: 92px 1fr 46px; gap: 10px; align-items: center; font-size: 14.5px; }
  .bar-row b { font-weight: 600; }
  .track { height: 10px; background: var(--field); border: 1px solid var(--line); border-radius: 999px; overflow: hidden; }
  .fill { height: 100%; width: 0; border-radius: 999px; transition: width 0.7s cubic-bezier(.2,.8,.2,1); }
  .bar-row .pc { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }

  .why { margin-top: 22px; padding-top: 18px; border-top: 1px solid var(--line); }
  .why h3 { margin: 0 0 10px; font-size: 15px; font-weight: 700; }
  .why ol { margin: 0; padding-left: 22px; color: var(--muted); font-size: 14.5px; }
  .why li + li { margin-top: 6px; }
  .why li b { color: var(--ink); font-weight: 700; }
  .error { color: #B3261E; font-weight: 600; margin: 6px 0 0; }

  .chart-grid { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(0, 1fr); gap: 28px; align-items: center; }
  #chart svg { display: block; width: 100%; height: auto; }
  #chart text { font-family: inherit; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 10px; font-size: 14px; color: var(--muted); }
  .legend span { display: inline-flex; align-items: center; gap: 7px; }
  .chart-text p { margin: 0 0 12px; color: var(--muted); }
  .chart-text p:last-child { margin-bottom: 0; }

  .about p { margin: 0 0 12px; color: var(--muted); max-width: 72ch; }
  .about p:last-child { margin-bottom: 0; }
  .about strong { color: var(--ink); }

  footer { margin-top: 26px; font-size: 13.5px; color: var(--muted); }

  @media (max-width: 880px) {
    .layout, .chart-grid { grid-template-columns: 1fr; }
    .result { position: static; min-height: 0; }
  }
  @media (max-width: 480px) {
    .card { padding: 22px 18px; }
    .mark { width: 54px; height: 54px; }
    .bar-row { grid-template-columns: 80px 1fr 44px; }
  }
  @media (prefers-reduced-motion: reduce) { .fill { transition: none; } }
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <svg class="mark" viewBox="0 0 64 64" aria-hidden="true">
      <path d="M32 62 C32 50 31 46 32 38" stroke="#1F9D8B" stroke-width="3.5" fill="none" stroke-linecap="round"/>
      <g transform="translate(32 40)">
        <path d="M0 0 C-17 -6 -24 -22 -16 -32 C-8 -30 -1 -16 0 0Z" fill="#8F7AE3"/>
        <path d="M0 0 C17 -6 24 -22 16 -32 C8 -30 1 -16 0 0Z" fill="#8F7AE3"/>
        <path d="M0 0 C-10 -12 -9 -30 0 -37 C9 -30 10 -12 0 0Z" fill="#5B3FC4"/>
        <path d="M0 -4 L0 -22" stroke="#F2C14E" stroke-width="3" stroke-linecap="round"/>
      </g>
    </svg>
    <div>
      <h1>Iris Species Classifier</h1>
      <p>Enter a flower's four measurements and the model will tell you which of three iris species it most likely is.</p>
    </div>
  </header>

  <div class="layout">
    <form id="form" class="card" novalidate>
      <h2>Flower measurements</h2>
      <div class="examples">
        <span class="lbl">Try an example</span>
        <button type="button" class="chip" data-ex="setosa"><i class="dot setosa"></i>Setosa</button>
        <button type="button" class="chip" data-ex="versicolor"><i class="dot versicolor"></i>Versicolor</button>
        <button type="button" class="chip" data-ex="virginica"><i class="dot virginica"></i>Virginica</button>
      </div>

      <div class="meas">
        <div class="meas-head"><label for="sepal_length">Sepal length</label><div class="num"><input id="sepal_length" type="number" step="0.1" min="4" max="8.5" required><span>cm</span></div></div>
        <input id="sepal_length_range" type="range" step="0.1" min="4" max="8.5" aria-label="Sepal length slider">
        <div class="ends"><span>4.0</span><span>8.5</span></div>
      </div>
      <div class="meas">
        <div class="meas-head"><label for="sepal_width">Sepal width</label><div class="num"><input id="sepal_width" type="number" step="0.1" min="1.5" max="4.8" required><span>cm</span></div></div>
        <input id="sepal_width_range" type="range" step="0.1" min="1.5" max="4.8" aria-label="Sepal width slider">
        <div class="ends"><span>1.5</span><span>4.8</span></div>
      </div>
      <div class="meas">
        <div class="meas-head"><label for="petal_length">Petal length</label><div class="num"><input id="petal_length" type="number" step="0.1" min="0.5" max="7.5" required><span>cm</span></div></div>
        <input id="petal_length_range" type="range" step="0.1" min="0.5" max="7.5" aria-label="Petal length slider">
        <div class="ends"><span>0.5</span><span>7.5</span></div>
      </div>
      <div class="meas">
        <div class="meas-head"><label for="petal_width">Petal width</label><div class="num"><input id="petal_width" type="number" step="0.1" min="0.1" max="3" required><span>cm</span></div></div>
        <input id="petal_width_range" type="range" step="0.1" min="0.1" max="3" aria-label="Petal width slider">
        <div class="ends"><span>0.1</span><span>3.0</span></div>
      </div>

      <div class="actions">
        <button type="submit" class="primary" id="predict-btn">Classify flower</button>
        <button type="button" class="ghost" id="reset-btn">Reset</button>
      </div>
    </form>

    <section class="card result" aria-labelledby="res-title">
      <h2 id="res-title">Prediction</h2>
      <div id="result" aria-live="polite">
        <p class="empty">Pick an example or enter your own measurements, then select Classify flower.</p>
      </div>
    </section>
  </div>

  <section class="card wide">
    <h2>Where the flower sits among the 150 in the dataset</h2>
    <div class="chart-grid">
      <div>
        <div id="chart"></div>
        <div class="legend">
          <span><i class="dot setosa"></i>Setosa</span>
          <span><i class="dot versicolor"></i>Versicolor</span>
          <span><i class="dot virginica"></i>Virginica</span>
        </div>
      </div>
      <div class="chart-text">
        <p>Each dot is a real flower from the dataset, plotted by petal length and petal width. The three species form clear groups.</p>
        <p>Setosa sits alone in the bottom left. Versicolor and virginica touch in the middle, and that overlap is where the model occasionally gets it wrong.</p>
        <p>After you classify a flower, a ringed marker shows where it lands.</p>
      </div>
    </div>
  </section>

  <section class="card wide about">
    <h2>About this model</h2>
    <p>It is a <strong>decision tree</strong> limited to three levels. It asks up to three yes/no questions about the measurements, for example whether petal length is above 2.45 cm, and the answers lead to one of the three species.</p>
    <p id="about-stats"></p>
  </section>

  <footer>A classroom project built on the classic Iris dataset (150 flowers, 50 per species).</footer>
</div>

<script>
  const BOOT = __BOOT__;
  const $ = (id) => document.getElementById(id);
  const FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"];
  const LABELS = { sepal_length: "Sepal length", sepal_width: "Sepal width", petal_length: "Petal length", petal_width: "Petal width" };
  const NAMES = { setosa: "Iris setosa", versicolor: "Iris versicolor", virginica: "Iris virginica" };
  const COLORS = { setosa: "#1F9D8B", versicolor: "#6B4FD3", virginica: "#E0832B" };
  const BLURB = {
    setosa: "Small petals. It is the easiest of the three species to tell apart.",
    versicolor: "Medium-sized petals, sitting between the other two species.",
    virginica: "Large petals. Usually the biggest flowers of the three."
  };
  const DEFAULTS = { sepal_length: 5.9, sepal_width: 3.0, petal_length: 5.1, petal_width: 1.8 };
  const EXAMPLES = {
    setosa: { sepal_length: 5.0, sepal_width: 3.4, petal_length: 1.5, petal_width: 0.2 },
    versicolor: { sepal_length: 5.9, sepal_width: 2.8, petal_length: 4.3, petal_width: 1.3 },
    virginica: { sepal_length: 6.6, sepal_width: 3.0, petal_length: 5.6, petal_width: 2.0 }
  };

  function setValues(v) {
    for (const f of FEATURES) {
      $(f).value = Number(v[f]).toFixed(1);
      $(f + "_range").value = v[f];
    }
  }

  function getValues() {
    const v = {};
    for (const f of FEATURES) v[f] = parseFloat($(f).value);
    return v;
  }

  // keep each number box and its slider in step
  for (const f of FEATURES) {
    $(f + "_range").addEventListener("input", (e) => { $(f).value = e.target.value; });
    $(f).addEventListener("input", (e) => {
      const n = parseFloat(e.target.value);
      if (!isNaN(n)) $(f + "_range").value = n;
    });
  }

  function drawChart(marker) {
    const W = 520, H = 330, L = 50, R = 16, T = 14, B = 46;
    const xmin = 0.5, xmax = 7.5, ymin = 0, ymax = 2.8;
    const sx = (v) => L + (v - xmin) / (xmax - xmin) * (W - L - R);
    const sy = (v) => H - B - (v - ymin) / (ymax - ymin) * (H - T - B);
    let s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Scatter plot of petal length against petal width for 150 iris flowers">';
    for (let v = 1; v <= 7; v++) {
      s += '<line x1="' + sx(v) + '" x2="' + sx(v) + '" y1="' + T + '" y2="' + (H - B) + '" stroke="#DDD8EC" stroke-width="1"/>';
      s += '<text x="' + sx(v) + '" y="' + (H - B + 18) + '" text-anchor="middle" font-size="12" fill="#5F5880">' + v + "</text>";
    }
    for (let v = 0; v <= 2.5; v += 0.5) {
      s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + sy(v) + '" y2="' + sy(v) + '" stroke="#DDD8EC" stroke-width="1"/>';
      s += '<text x="' + (L - 8) + '" y="' + (sy(v) + 4) + '" text-anchor="end" font-size="12" fill="#5F5880">' + v.toFixed(1) + "</text>";
    }
    s += '<text x="' + ((L + W - R) / 2) + '" y="' + (H - 8) + '" text-anchor="middle" font-size="13" font-weight="600" fill="#1F1B3D">Petal length (cm)</text>';
    s += '<text transform="translate(14 ' + ((T + H - B) / 2) + ') rotate(-90)" text-anchor="middle" font-size="13" font-weight="600" fill="#1F1B3D">Petal width (cm)</text>';
    for (const p of BOOT.points) {
      s += '<circle cx="' + sx(p[0]).toFixed(1) + '" cy="' + sy(p[1]).toFixed(1) + '" r="4" fill="' + COLORS[BOOT.classes[p[2]]] + '" fill-opacity="0.6"/>';
    }
    if (marker) {
      const x = Math.min(Math.max(sx(marker.pl), L + 6), W - R - 6);
      const y = Math.min(Math.max(sy(marker.pw), T + 6), H - B - 6);
      s += '<circle cx="' + x + '" cy="' + y + '" r="12" fill="none" stroke="#1F1B3D" stroke-width="3"/>';
      s += '<circle cx="' + x + '" cy="' + y + '" r="5.5" fill="' + COLORS[marker.species] + '"/>';
      const ty = y < T + 34 ? y + 30 : y - 19;
      s += '<text x="' + x + '" y="' + ty + '" text-anchor="middle" font-size="13" font-weight="700" fill="#1F1B3D" stroke="#fff" stroke-width="4" paint-order="stroke">Your flower</text>';
    }
    $("chart").innerHTML = s + "</svg>";
  }

  function pct(p) { return Math.round(p * 100) + "%"; }

  function renderResult(r) {
    const best = r.prediction;
    const top = r.probabilities[best];
    const note = top >= 0.9
      ? "The model is confident about this one."
      : top >= 0.7
        ? "The model is fairly sure, but this flower is close to another species."
        : "This flower sits near the boundary between two species, so the model is unsure.";
    let html = '<div class="verdict"><i class="dot ' + best + '"></i><strong>' + NAMES[best] + "</strong></div>" +
      '<p class="about-species">' + BLURB[best] + "</p>" +
      '<p class="conf">' + pct(top) + " confidence. " + note + "</p><div class=\"bars\">";
    for (const c of BOOT.classes) {
      html += '<div class="bar-row"><b>' + c.charAt(0).toUpperCase() + c.slice(1) + '</b><div class="track"><div class="fill" data-w="' + (r.probabilities[c] * 100) + '" style="background:' + COLORS[c] + '"></div></div><span class="pc">' + pct(r.probabilities[c]) + "</span></div>";
    }
    html += '</div><div class="why"><h3>How the tree decided</h3><ol>';
    for (const s of r.path) {
      html += "<li><b>" + LABELS[s.feature] + " is " + s.value + " cm</b>, which is " + (s.left ? "at most " : "above ") + s.threshold + " cm</li>";
    }
    html += "</ol></div>";
    $("result").innerHTML = html;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      document.querySelectorAll(".fill").forEach((el) => { el.style.width = el.dataset.w + "%"; });
    }));
  }

  function showEmpty() {
    $("result").innerHTML = '<p class="empty">Pick an example or enter your own measurements, then select Classify flower.</p>';
    drawChart(null);
  }

  async function classify() {
    const form = $("form");
    if (!form.checkValidity()) { form.reportValidity(); return; }
    const v = getValues();
    const btn = $("predict-btn");
    btn.disabled = true; btn.textContent = "Classifying\u2026";
    try {
      const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(v) });
      if (!res.ok) throw new Error("bad response");
      const r = await res.json();
      renderResult(r);
      drawChart({ pl: v.petal_length, pw: v.petal_width, species: r.prediction });
    } catch (err) {
      $("result").innerHTML = '<p class="error">The prediction could not be made. Check that every value is within its range and try again.</p>';
    } finally {
      btn.disabled = false; btn.textContent = "Classify flower";
    }
  }

  $("form").addEventListener("submit", (e) => { e.preventDefault(); classify(); });

  document.querySelectorAll("[data-ex]").forEach((b) => {
    b.addEventListener("click", () => { setValues(EXAMPLES[b.dataset.ex]); classify(); });
  });

  $("reset-btn").addEventListener("click", () => { setValues(DEFAULTS); showEmpty(); });

  $("about-stats").innerHTML = "It learned from <strong>" + BOOT.train_size + " flowers</strong> and was then tested on <strong>" + BOOT.test_size +
    "</strong> it had never seen, getting <strong>" + (BOOT.accuracy * 100).toFixed(1) + "%</strong> of them right.";

  setValues(DEFAULTS);
  drawChart(null);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE.replace("__BOOT__", json.dumps(BOOT))
