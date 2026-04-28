  // FIELD_MAP is defined in each form html page because the mapping
  // is slightly different between linear and nonlinear.
  // This function reads window.FIELD_MAP set by the page.
  async function lookupParcel() {
    const input = document.getElementById('parcel-input');
    const status = document.getElementById('parcel-status');
    if (!input || !status) return;

    const id = input.value.trim();
    if (!id) { status.textContent = 'NO ID ENTERED'; status.className = 'err'; return; }
    status.textContent = 'SEARCHING...'; status.className = '';

    try {
      const res = await fetch(`/parcel/${encodeURIComponent(id)}`);
      if (!res.ok) { status.textContent = 'NOT FOUND'; status.className = 'err'; return; }
      const data = await res.json();

      let filled = 0;
      const fieldMap = window.FIELD_MAP   || {};
      const totalFields = window.TOTAL_FIELDS || Object.keys(fieldMap).length;

      for (const [fieldKey, csvCol] of Object.entries(fieldMap)) {
        const el = document.getElementById(`f_${fieldKey}`);
        if (!el) continue;
        const val = data[csvCol.toLowerCase()];
        if (val !== null && val !== undefined) {
          if (el.tagName === 'SELECT') {
            const strVal = String(val).trim();
            // Try exact match first, then anyt different thing
            const opt = [...el.options].find(o => o.value === strVal)
                    || [...el.options].find(o => o.value.toLowerCase() === strVal.toLowerCase());
            if (opt) { el.value = opt.value; el.closest('.field').classList.add('filled'); filled++;
            }
          } else {
            // Strip dollar signs and commas for number inputs
            let cleanVal = String(val).replace(/[$,\s]/g, '');
            el.value = cleanVal;
            el.classList.add('autofilled');
            el.closest('.field').classList.add('filled');
            filled++;
          }
        }
      }

      // Populate hidden nbhd_code2 — try all likely column name variants
      const hiddenNbhd = document.getElementById('hidden-nbhd-code2');
      if (hiddenNbhd) {
        const nbhdVal = data['nbhdcode2'] ?? data['nbhd_code2'] ?? data['nbhdcode']
                    ?? data['nbhd_code'] ?? data['neighborhoodcode2'] ?? data['neighborhood_code2'] ?? '';
        hiddenNbhd.value = String(nbhdVal === null ? '' : nbhdVal).trim();
      }

      status.textContent = `${filled}/${totalFields} FILLED`;
      status.className   = filled > 0 ? 'ok' : 'err';
      const countEl = document.getElementById('feature-count');
      if (countEl) countEl.textContent = `${filled} FEATURES AUTO-FILLED`;
    } catch (e) {
      status.textContent = 'ERROR'; status.className = 'err';
    }
  }

  // Wire up parcel input events if the element exists
  (function initParcelInput() {
    const input = document.getElementById('parcel-input');
    if (!input) return;

    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); lookupParcel(); }
    });

    // Page sets window.PARCEL_INPUT_PATTERN to override.
    input.addEventListener('input', function () {
      const pattern = window.PARCEL_INPUT_PATTERN || /[^A-Za-z0-9-]/g;
      this.value = this.value.replace(pattern, '');
    });
  })();

  /* PARCEL CSV UPLOAD  (main.html, main_nonlinear.html) */
  (function checkParcelStatus() {
    const label = document.getElementById('upload-label');
    const status = document.getElementById('upload-status');
    if (!label || !status) return;

    fetch('/parcel-status')
      .then(r => r.json())
      .then(data => {
        if (data.loaded) {
          status.textContent = `LOADED ${data.rows.toLocaleString()} ROWS · ${data.columns} COLS`;
          status.className = 'ok';
          label.textContent = '✓ PARCEL DATA READY';
          label.className = 'upload-banner-label';
        }
      })
      .catch(() => {}); // silently ignore errors, user can just upload if status check fails
  })();

  (function initParcelUpload() {
    const fileInput = document.getElementById('parcel-file-input');
    if (!fileInput) return;

    fileInput.addEventListener('change', async function () {
      const file = this.files[0];
      if (!file) return;
      const status = document.getElementById('upload-status');
      const label = document.getElementById('upload-label');
      status.textContent = 'UPLOADING...'; status.className = '';

      const fd = new FormData();
      fd.append('file', file);
      try {
        const res = await fetch('/upload-parcel-data', { method: 'POST', body: fd });
        const data = await res.json();
        if (res.ok && data.status === 'ok') {
          status.textContent = `LOADED ${data.rows.toLocaleString()} ROWS · ${data.columns.length} COLS`;
          status.className = 'ok';
          label.textContent = '✓ PARCEL DATA READY';
          label.className = 'upload-banner-label';
        } else {
          status.textContent = data.detail || 'Upload failed'; status.className = 'err';
        }
      } catch (e) {
        status.textContent = 'UPLOAD ERROR'; status.className = 'err';
      }
    });
  })();

  /* HISTORY PAGE - Gotts be able to return to a result if clicked */
  function toggleFeatures(id) {
    const row = document.getElementById('features-' + id);
    const iconCell = document.getElementById('icon-' + id);
    if (!row || !iconCell) return;

    const isOpen = row.classList.contains('open');
    document.querySelectorAll('.feature-row').forEach(r => r.classList.remove('open'));
    document.querySelectorAll('.data-row').forEach(r => r.classList.remove('open-indicator'));

    if (!isOpen) {
      row.classList.add('open');
      iconCell.closest('tr').classList.add('open-indicator');
    }
  }

  /* RESULT PAGE — diagnostic plots + waterfall charts
    All plots and charts are stopped if the required data 
    aren't passed to prevent errors. */
  (function initResultCharts() {
    // Data injected by Jinja lives on window.
    if (typeof PREDICTED === 'undefined') return;

    /* Using raw DOM API for maximum control and minimal dependencies. */
    const NS = 'http://www.w3.org/2000/svg';

    function mkSvg(w, h) {
      const s = document.createElementNS(NS, 'svg');
      s.setAttribute('viewBox', `0 0 ${w} ${h}`);
      s.setAttribute('width', w); s.setAttribute('height', h);
      return s;
    }

    function el(tag, attrs) {
      const e = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
      return e;
    }

    function tx(x, y, str, opts = {}) {
      const t = el('text', { x, y, fill: '#555', 'font-family': "'Share Tech Mono',monospace", 'font-size': '8', 'text-anchor': 'middle', ...opts });
      t.textContent = str;
      return t;
    }

    function makeScales(xs, ys, W, H, padL = 36, padR = 12, padT = 10, padB = 26) {
      const xMin = Math.min(...xs), xMax = Math.max(...xs);
      const yMin = Math.min(...ys), yMax = Math.max(...ys);
      const xPad = (xMax - xMin) * 0.04 || 1;
      const yPad = (yMax - yMin) * 0.04 || 1;
      const sx = v => padL + (v - xMin - xPad / 2) / ((xMax - xMin + xPad) || 1) * (W - padL - padR);
      const sy = v => H - padB - (v - yMin - yPad / 2) / ((yMax - yMin + yPad) || 1) * (H - padT - padB);
      return { sx, sy, xMin, xMax, yMin, yMax, padL, padR, padT, padB, W, H };
    }

    function axes(sc) {
      const g = el('g', {});
      g.appendChild(el('line', { x1: sc.padL, y1: sc.H - sc.padB, x2: sc.W - sc.padR, y2: sc.H - sc.padB, stroke: '#2a2a2a', 'stroke-width': '1' }));
      g.appendChild(el('line', { x1: sc.padL, y1: sc.padT, x2: sc.padL, y2: sc.H - sc.padB, stroke: '#2a2a2a', 'stroke-width': '1' }));
      for (let i = 0; i <= 3; i++) {
        const v = sc.xMin + (sc.xMax - sc.xMin) * i / 3;
        const x = sc.sx(v);
        g.appendChild(el('line', { x1: x, y1: sc.H - sc.padB, x2: x, y2: sc.H - sc.padB + 3, stroke: '#333', 'stroke-width': '1' }));
        g.appendChild(tx(x, sc.H - sc.padB + 11, v >= 100000 ? '$' + Math.round(v / 1000) + 'k' : v.toFixed(v > 10 ? 0 : 2)));
      }
      for (let i = 0; i <= 3; i++) {
        const v = sc.yMin + (sc.yMax - sc.yMin) * i / 3;
        const y = sc.sy(v);
        g.appendChild(el('line', { x1: sc.padL - 3, y1: y, x2: sc.padL, y2: y, stroke: '#333', 'stroke-width': '1' }));
        g.appendChild(tx(sc.padL - 5, y + 3, v >= 100000 ? '$' + Math.round(v / 1000) + 'k' : v.toFixed(v > 10 ? 0 : 2), { 'text-anchor': 'end' }));
      }
      return g;
    }

    const W = 260, H = 190;

    /* ── 1. Predicted vs Actual ── */
    function plotActualVsPred(d) {
      const svg = mkSvg(W, H);
      const actDollar = d.actual.map(v => Math.exp(v));
      const predDollar = d.predicted.map(v => Math.exp(v));
      const sc = makeScales(actDollar, predDollar, W, H);
      svg.appendChild(axes(sc));
      const mn = Math.max(sc.xMin, sc.yMin), mx = Math.min(sc.xMax, sc.yMax);
      svg.appendChild(el('line', { x1: sc.sx(mn), y1: sc.sy(mn), x2: sc.sx(mx), y2: sc.sy(mx), stroke: '#e8a020', 'stroke-width': '1.2', 'stroke-dasharray': '5,3' }));
      actDollar.forEach((xi, i) => {
        const over = predDollar[i] > xi;
        svg.appendChild(el('circle', { cx: sc.sx(xi), cy: sc.sy(predDollar[i]), r: '1.5', fill: over ? '#2ecc71' : '#e74c3c', opacity: '0.4' }));
      });
      svg.appendChild(el('rect', { x: sc.padL + 4, y: sc.padT + 4,  width: 7, height: 7, fill: '#2ecc71', rx: '1' }));
      svg.appendChild(tx(sc.padL + 14, sc.padT + 10, 'Over',  { 'text-anchor': 'start', 'font-size': '7', fill: '#888' }));
      svg.appendChild(el('rect', { x: sc.padL + 4, y: sc.padT + 16, width: 7, height: 7, fill: '#e74c3c', rx: '1' }));
      svg.appendChild(tx(sc.padL + 14, sc.padT + 22, 'Under', { 'text-anchor': 'start', 'font-size': '7', fill: '#888' }));
      return svg;
    }

    /* ── 2. Residuals vs Fitted ── */
    function plotResidVsFitted(d) {
      const svg = mkSvg(W, H);
      const sc = makeScales(d.x, d.y, W, H);
      svg.appendChild(axes(sc));

      // Zero reference line
      svg.appendChild(el('line', {
        x1: sc.padL, y1: sc.sy(0), x2: sc.W - sc.padR, y2: sc.sy(0),
        stroke: '#e8a020', 'stroke-width': '1', 'stroke-dasharray': '5,3'
      }));

      // Points
      d.x.forEach((xi, i) => {
        const resid = d.y[i];
        const fill = resid > 0 ? '#2ecc71' : '#e74c3c';
        svg.appendChild(el('circle', {
          cx: sc.sx(xi), cy: sc.sy(resid),
          r: '1.4', fill, opacity: '0.35'
        }));
      });

      // Labels
      svg.appendChild(tx(sc.W / 2, sc.H - 2, 'Fitted (log)', { 'font-size': '7', fill: '#444' }));

      return svg;
    }

      /* ── 3. Feature Influence bar chart ── */
    function plotFeatureInfluence() {
      const items = RAW
        .map(r => ({ name: r[0], dollar: r[1] }))
        .filter(r => r.dollar !== 0)
        .sort((a, b) => Math.abs(b.dollar) - Math.abs(a.dollar))
        .slice(0, 10);

      const BAR_H = 13, GAP = 5, PAD_L = 82, PAD_R = 52, PAD_T = 10, PAD_B = 20;
      const svgH = PAD_T + items.length * (BAR_H + GAP) + PAD_B;
      const svg = mkSvg(W, svgH);
      const chartW = W - PAD_L - PAD_R;
      const maxAbs = Math.max(...items.map(r => Math.abs(r.dollar)));
      const zx = PAD_L + chartW / 2;

      svg.appendChild(el('line', { x1: zx, y1: PAD_T - 4, x2: zx, y2: svgH - PAD_B + 4, stroke: '#333', 'stroke-width': '1' }));

      const fmt = v => {
        const abs = Math.abs(v);
        return (v < 0 ? '−' : v > 0 ? '+' : '') + (abs >= 1000 ? '$' + Math.round(abs / 1000) + 'k' : '$' + Math.round(abs));
      };
      [[-maxAbs, PAD_L], [0, zx], [maxAbs, PAD_L + chartW]].forEach(([v, x]) => {
        svg.appendChild(tx(x, svgH - PAD_B + 10, fmt(v), { 'font-size': '7', fill: '#444' }));
      });

      items.forEach((item, i) => {
        const y = PAD_T + i * (BAR_H + GAP);
        const pos = item.dollar >= 0;
        const barW = Math.max((Math.abs(item.dollar) / maxAbs) * (chartW / 2), 1);
        const barX = pos ? zx : zx - barW;
        const fill = pos ? '#2ecc71' : '#e74c3c';

        svg.appendChild(el('rect', { x: barX, y, width: barW, height: BAR_H, fill, rx: '1', opacity: '0.82' }));

        const maxChars = 13;
        const label = item.name.length > maxChars ? item.name.slice(0, maxChars - 1) + '…' : item.name;
        svg.appendChild(tx(PAD_L - 5, y + BAR_H / 2 + 3, label, { 'text-anchor': 'end', 'font-size': '7.5', fill: '#aaa' }));

        const dollarStr = (pos ? '+$' : '−$') + Math.round(Math.abs(item.dollar)).toLocaleString();
        const labelX = pos ? zx + barW + 3 : zx - barW - 3;
        svg.appendChild(tx(labelX, y + BAR_H / 2 + 3, dollarStr, { 'text-anchor': pos ? 'start' : 'end', 'font-size': '7', fill: fill }));
      });
      return svg;
    }

    /* ── 4. Residual Distribution histogram ── */
    function plotResidHist(d) {
      const vals = d.values;
      const BINS = 28;
      const min = Math.min(...vals), max = Math.max(...vals);
      const bw = (max - min) / BINS;
      const counts = new Array(BINS).fill(0);
      vals.forEach(v => { const b = Math.min(Math.floor((v - min) / bw), BINS - 1); counts[b]++; });

      const padL = 28, padR = 8, padT = 10, padB = 22;
      const chartW = W - padL - padR, chartH = H - padT - padB;
      const maxC = Math.max(...counts);
      const bpx = chartW / BINS;
      const svg = mkSvg(W, H);

      // Axes
      svg.appendChild(el('line', { x1: padL, y1: H - padB, x2: W - padR, y2: H - padB, stroke: '#2a2a2a', 'stroke-width': '1' }));
      svg.appendChild(el('line', { x1: padL, y1: padT,     x2: padL,     y2: H - padB, stroke: '#2a2a2a', 'stroke-width': '1' }));

      // Bars — purple, coloured by distance from zero
      counts.forEach((c, i) => {
        const bh = (c / maxC) * chartH;
        const binMid = min + (i + 0.5) * bw;
        // Fade from amber (near zero) to purple (in tails)
        const dist = Math.abs(binMid) / (Math.max(Math.abs(min), Math.abs(max)) || 1);
        const fill = dist < 0.35 ? '#e8a020' : '#a78bfa';
        svg.appendChild(el('rect', {
          x: padL + i * bpx + 0.5,
          y: H - padB - bh,
          width: Math.max(bpx - 1, 1),
          height: bh,
          fill, opacity: '0.75'
        }));
      });

      // Zero line
      const zx = padL + ((0 - min) / ((max - min) || 1)) * chartW;
      if (zx > padL && zx < W - padR) {
        svg.appendChild(el('line', { x1: zx, y1: padT, x2: zx, y2: H - padB, stroke: '#e74c3c', 'stroke-width': '1', 'stroke-dasharray': '4,3' }));
      }

      // x tick labels
      for (let i = 0; i <= 2; i++) {
        const v = min + (max - min) * i / 2;
        const x = padL + (v - min) / ((max - min) || 1) * chartW;
        svg.appendChild(tx(x, H - padB + 11, v.toFixed(2), { 'font-size': '7' }));
      }

      // Normal curve overlay — estimate mean and sigma from data, then plot normal PDF scaled to match histogram
      const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
      const sigma = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length) || 1;
      const totalN = vals.length;
      const pathPts = [];
      for (let px = padL; px <= W - padR; px += 2) {
        const v = min + ((px - padL) / chartW) * (max - min);
        const pdf = Math.exp(-0.5 * ((v - mean) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
        const freq = pdf * totalN * bw;
        const py = H - padB - (freq / maxC) * chartH;
        pathPts.push((px === padL ? 'M' : 'L') + px + ',' + Math.max(padT, py));
      }
      svg.appendChild(el('path', { d: pathPts.join(' '), stroke: '#38bdf8', 'stroke-width': '1.2', fill: 'none', opacity: '0.7' }));

      // Legend
      svg.appendChild(el('line', { x1: padL + 4, y1: padT + 8, x2: padL + 14, y2: padT + 8, stroke: '#38bdf8', 'stroke-width': '1.2' }));
      svg.appendChild(tx(padL + 16, padT + 11, 'Normal', { 'text-anchor': 'start', 'font-size': '7', fill: '#888' }));

      return svg;
    }

    /* Render diagnostic grid (linear only) */
    const grid = document.getElementById('diag-grid');
    if (grid && typeof LIN_DIAG !== 'undefined' && LIN_DIAG) {
      const panels = [
        { key: 'actual_vs_pred', title: 'Predicted vs Actual ($)', fn: plotActualVsPred },
        { key: 'resid_vs_fitted', title: 'Residuals vs Fitted',     fn: plotResidVsFitted },
        { key: null, title: 'Feature Influence ($)',    fn: plotFeatureInfluence },
        { key: 'resid_hist', title: 'Residual Distribution',   fn: plotResidHist },
      ];
      panels.forEach(p => {
        const cell = document.createElement('div');
        cell.className = 'diag-cell';
        const titleEl = document.createElement('div');
        titleEl.className = 'diag-title';
        titleEl.textContent = p.title;
        cell.appendChild(titleEl);
        try {
          cell.appendChild(p.key === null ? p.fn() : p.fn(LIN_DIAG[p.key]));
        } catch (e) {
          titleEl.textContent += ' (no data)';
          console.error(e);
        }
        grid.appendChild(cell);
      });
    }

    /* Render waterfall chart (nonlinear only) */
    const waterfallSvg = document.getElementById('waterfall-svg');
    if (waterfallSvg && typeof LIN_DIAG !== 'undefined' && LIN_DIAG === null) {
      const contribs = RAW.map(r => ({ name: r[0], dollar: r[1] }))
        .filter(r => r.dollar !== 0)
        .sort((a, b) => Math.abs(b.dollar) - Math.abs(a.dollar));

      const TOP_N = 10;
      const shown = contribs.slice(0, TOP_N);
      const rest = contribs.slice(TOP_N);
      const restSum = rest.reduce((s, r) => s + r.dollar, 0);
      if (Math.abs(restSum) > 500) shown.push({ name: 'Other features', dollar: restSum });

      const steps = [];
      let running = BASELINE;
      steps.push({ label: 'Baseline', start: 0, end: BASELINE, isBaseline: true, delta: 0 });
      for (const { name, dollar } of shown) {
        steps.push({ label: name, start: running, end: running + dollar, delta: dollar });
        running += dollar;
      }
      steps.push({ label: 'Predicted Price', start: 0, end: PREDICTED, isTotal: true, delta: 0 });

      const BAR_H = 30, GAP = 5, PAD_L = 196, PAD_R = 120, PAD_T = 14, PAD_B = 28;
      const N = steps.length, wH = PAD_T + N * (BAR_H + GAP) + PAD_B;
      const W_MIN = 600;
      const avail = waterfallSvg.parentElement.clientWidth || W_MIN;
      const wW = Math.max(avail, W_MIN);
      const chartW = wW - PAD_L - PAD_R;

      waterfallSvg.setAttribute('viewBox', `0 0 ${wW} ${wH}`);
      waterfallSvg.setAttribute('width',  wW);
      waterfallSvg.setAttribute('height', wH);
      waterfallSvg.innerHTML = '';

      const allV = steps.flatMap(s => [s.start, s.end]);
      const minV = Math.min(...allV, PI_LO) * 0.97;
      const maxV = Math.max(...allV, PI_HI) * 1.03;
      const scale = v => PAD_L + (v - minV) / (maxV - minV) * chartW;

      const C_POS = '#2ecc71', C_NEG = '#e74c3c', C_BASE = '#e8a020', C_TOTAL = '#38bdf8';

      function el2(tag, attrs) {
        const e = document.createElementNS(NS, tag);
        for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
        return e;
      }
      function tx2(x, y, str, extra = {}) {
        const t = el2('text', { x, y, fill: '#c8c8c8', 'font-family': "'Share Tech Mono',monospace", 'font-size': '10', ...extra });
        t.textContent = str; return t;
      }
      const fmt = v => '$' + Math.round(Math.abs(v)).toLocaleString();
      const fmtD = v => (v >= 0 ? '+' : '−') + '$' + Math.round(Math.abs(v)).toLocaleString();

      for (let i = 0; i <= 5; i++) {
        const v = minV + (maxV - minV) * i / 5, x = scale(v);
        waterfallSvg.appendChild(el2('line', { x1: x, y1: PAD_T - 6, x2: x, y2: wH - PAD_B + 4, stroke: '#1e1e1e', 'stroke-width': '1' }));
        waterfallSvg.appendChild(tx2(x, wH - PAD_B + 14, v >= 1000 ? '$' + Math.round(v / 1000) + 'k' : '$' + Math.round(v), { fill: '#555', 'font-size': '9', 'text-anchor': 'middle' }));
      }

      const tY = PAD_T + (N - 1) * (BAR_H + GAP);
      waterfallSvg.appendChild(el2('rect', { x: scale(PI_LO), y: tY - 6, width: scale(PI_HI) - scale(PI_LO), height: BAR_H + 12, fill: 'rgba(56,189,248,.07)', stroke: 'rgba(56,189,248,.2)', 'stroke-width': '1', rx: '1' }));
      waterfallSvg.appendChild(tx2(scale((PI_LO + PI_HI) / 2), tY - 10, '← 95% PI →', { fill: 'rgba(56,189,248,.45)', 'font-size': '8', 'text-anchor': 'middle' }));

      steps.forEach((step, i) => {
        const y = PAD_T + i * (BAR_H + GAP);
        const lo = Math.min(step.start, step.end), hi = Math.max(step.start, step.end);
        const xL = scale(lo), xR = scale(hi), bW = Math.max(xR - xL, 2);
        let fill = step.delta >= 0 ? C_POS : C_NEG;
        if (step.isBaseline) fill = C_BASE;
        if (step.isTotal)    fill = C_TOTAL;

        if (i > 0 && !step.isTotal) {
          const pEnd = scale(steps[i - 1].end);
          waterfallSvg.appendChild(el2('line', { x1: pEnd, y1: y - GAP + 1, x2: pEnd, y2: y + 1, stroke: '#333', 'stroke-width': '1', 'stroke-dasharray': '3,2' }));
        }
        waterfallSvg.appendChild(el2('rect', { x: xL, y, width: bW, height: BAR_H, fill, rx: '1', opacity: '0.88' }));

        const maxC = 28;
        const label = step.label.length > maxC ? step.label.slice(0, maxC - 1) + '…' : step.label;
        waterfallSvg.appendChild(tx2(PAD_L - 8, y + BAR_H / 2 + 4, label, { 'text-anchor': 'end', fill: step.isBaseline ? C_BASE : step.isTotal ? C_TOTAL : '#c8c8c8', 'font-size': '10' }));

        const valStr = (step.isBaseline || step.isTotal) ? fmt(step.end) : fmtD(step.delta);
        waterfallSvg.appendChild(tx2(xR + 6, y + BAR_H / 2 + 4, valStr, { fill: step.isTotal ? C_TOTAL : step.isBaseline ? C_BASE : step.delta >= 0 ? C_POS : C_NEG, 'font-size': '10' }));
      });
    }

    /* Render SHAP Stack Waterfall (nonlinear only) */
    const stackSvg = document.getElementById('shap-stack-svg');
    if (stackSvg && typeof LIN_DIAG !== 'undefined' && LIN_DIAG === null) {
      const contribs = RAW
        .map(r => ({ name: r[0], dollar: r[1], featVal: r.length > 2 ? r[2] : null }))
        .filter(r => r.dollar !== 0)
        .sort((a, b) => Math.abs(b.dollar) - Math.abs(a.dollar));

      const TOP_N = 12;
      const shown = contribs.slice(0, TOP_N);
      const rest = contribs.slice(TOP_N);
      const restSum = rest.reduce((s, r) => s + r.dollar, 0);
      if (Math.abs(restSum) > 500) shown.push({ name: rest.length + ' other features', dollar: restSum, featVal: null });

      shown.sort((a, b) => b.dollar - a.dollar);

      const N = shown.length;
      const BAR_H = 22, GAP = 6, PAD_L = 210, PAD_R = 90, PAD_T = 36, PAD_B = 28;
      const svgH = PAD_T + N * (BAR_H + GAP) + PAD_B;
      const avail = stackSvg.parentElement.clientWidth || 700;
      const svgW = Math.max(avail, 660);
      const chartW = svgW - PAD_L - PAD_R;

      stackSvg.setAttribute('viewBox', `0 0 ${svgW} ${svgH}`);
      stackSvg.setAttribute('width',  svgW);
      stackSvg.setAttribute('height', svgH);
      stackSvg.innerHTML = '';

      function sel(tag, attrs) {
        const e = document.createElementNS(NS, tag);
        for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
        return e;
      }
      function stx(x, y, str, opts = {}) {
        const t = sel('text', { x, y, fill: '#c8c8c8', 'font-family': "'Share Tech Mono',monospace", 'font-size': '9', ...opts });
        t.textContent = str; return t;
      }

      const allEnds = [];
      let run = BASELINE;
      shown.forEach(r => { run += r.dollar; allEnds.push(run); });
      const allVals = [BASELINE, ...allEnds, PI_LO, PI_HI];
      const domMin = Math.min(...allVals) * 0.97;
      const domMax = Math.max(...allVals) * 1.03;
      const sx = v => PAD_L + (v - domMin) / (domMax - domMin) * chartW;
      const fmtK = v => '$' + (Math.abs(v) >= 1000 ? Math.round(v / 1000) + 'k' : Math.round(v));

      const bx = sx(BASELINE);
      stackSvg.appendChild(sel('line', { x1: bx, y1: PAD_T - 6, x2: bx, y2: svgH - PAD_B + 4, stroke: '#888', 'stroke-width': '1', 'stroke-dasharray': '4,3' }));
      stackSvg.appendChild(stx(bx, PAD_T - 10, 'f(x) = ' + fmtK(BASELINE), { 'text-anchor': 'middle', 'font-size': '9', fill: '#888' }));

      for (let i = 0; i <= 5; i++) {
        const v = domMin + (domMax - domMin) * i / 5, x = sx(v);
        stackSvg.appendChild(sel('line', { x1: x, y1: PAD_T - 2, x2: x, y2: svgH - PAD_B + 2, stroke: '#1e1e1e', 'stroke-width': '1' }));
        stackSvg.appendChild(stx(x, svgH - PAD_B + 12, fmtK(v), { 'text-anchor': 'middle', 'font-size': '8', fill: '#444' }));
      }

      const C_POS = '#e8547a', C_NEG = '#5096d6';
      let running = BASELINE;

      shown.forEach((item, i) => {
        const y = PAD_T + i * (BAR_H + GAP);
        const pos = item.dollar >= 0;
        const fill = pos ? C_POS : C_NEG;
        const barStart = running;
        const barEnd = running + item.dollar;
        const xL = Math.min(sx(barStart), sx(barEnd));
        const xR = Math.max(sx(barStart), sx(barEnd));
        const bW = Math.max(xR - xL, 2);

        if (i > 0) {
          stackSvg.appendChild(sel('line', { x1: sx(running), y1: y - GAP + 1, x2: sx(running), y2: y, stroke: '#333', 'stroke-width': '0.8', 'stroke-dasharray': '3,2' }));
        }
        stackSvg.appendChild(sel('rect', { x: xL, y, width: bW, height: BAR_H, fill, rx: '2', opacity: '0.9' }));

        const deltaStr = (Math.abs(item.dollar) >= 1000
          ? (pos ? '+' : '−') + '$' + Math.round(Math.abs(item.dollar) / 1000) + 'k'
          : (pos ? '+' : '−') + '$' + Math.round(Math.abs(item.dollar)));
        const labelInside = bW > 36;
        const labelX = labelInside ? (xL + xR) / 2 : pos ? xR + 4 : xL - 4;
        stackSvg.appendChild(stx(labelX, y + BAR_H / 2 + 3, deltaStr, { 'text-anchor': labelInside ? 'middle' : pos ? 'start' : 'end', 'font-size': '8', fill: labelInside ? '#fff' : fill }));

        const maxChars = 28;
        const name = item.name.length > maxChars ? item.name.slice(0, maxChars - 1) + '…' : item.name;
        stackSvg.appendChild(stx(PAD_L - 8, y + BAR_H / 2 + 3, name, { 'text-anchor': 'end', 'font-size': '8.5', fill: '#aaa' }));

        const cumX = sx(barEnd);
        stackSvg.appendChild(stx(pos ? cumX + 4 : xL - 4, y + BAR_H / 2 + 3, fmtK(barEnd), { 'text-anchor': pos ? 'start' : 'end', 'font-size': '7.5', fill: '#555' }));

        running = barEnd;
      });

      const px = sx(PREDICTED);
      stackSvg.appendChild(sel('line', { x1: px, y1: PAD_T, x2: px, y2: svgH - PAD_B, stroke: '#e8a020', 'stroke-width': '1.2', 'stroke-dasharray': '5,3', opacity: '0.7' }));
      stackSvg.appendChild(stx(px, svgH - PAD_B + 20, fmtK(PREDICTED), { 'text-anchor': 'middle', 'font-size': '8.5', fill: '#e8a020' }));

      const lastY = PAD_T + (N - 1) * (BAR_H + GAP);
      stackSvg.appendChild(sel('rect', { x: sx(PI_LO), y: lastY - 4, width: sx(PI_HI) - sx(PI_LO), height: BAR_H + 8, fill: 'rgba(232,160,32,0.07)', stroke: 'rgba(232,160,32,0.25)', 'stroke-width': '1', rx: '1' }));
    }

  })();