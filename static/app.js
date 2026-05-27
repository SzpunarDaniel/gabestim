// AutoEstim — app.js (Playwright + ML)

const brandSel  = document.getElementById('brand');
const modelSel  = document.getElementById('model');
const form      = document.getElementById('estimForm');
const submitBtn = document.getElementById('submitBtn');
const loader    = document.getElementById('loader');
const errorBox  = document.getElementById('errorBox');
const errorMsg  = document.getElementById('errorMsg');
const results   = document.getElementById('results');

let priceChart = null;
let deprecChart = null;

// ── Chargement des modèles ──
brandSel.addEventListener('change', async () => {
  const brand = brandSel.value;
  modelSel.innerHTML = '<option value="">Chargement…</option>';
  modelSel.disabled = true;
  if (!brand) { modelSel.innerHTML = '<option value="">Choisir une marque d\'abord…</option>'; return; }
  try {
    const res = await fetch(`/api/models/${encodeURIComponent(brand)}`);
    const models = await res.json();
    modelSel.innerHTML = '<option value="">Sélectionner…</option>';
    models.forEach(m => { const o = document.createElement('option'); o.value = m; o.textContent = m; modelSel.appendChild(o); });
    modelSel.disabled = false;
  } catch (e) { modelSel.innerHTML = '<option value="">Erreur de chargement</option>'; }
});

// ── Soumission ──
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  startSearch();
  const payload = {
    brand:        brandSel.value,
    model:        modelSel.value,
    year_min:     document.getElementById('year_min').value,
    year_max:     document.getElementById('year_max').value,
    mileage_min:  document.getElementById('mileage_min').value,
    mileage_max:  document.getElementById('mileage_max').value,
    fuel:         document.getElementById('fuel').value,
    transmission: document.getElementById('transmission').value,
  };
  try {
    const res  = await fetch('/api/estimate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await res.json();
    endSearch();
    if (!res.ok) { showError(data.error || 'Une erreur est survenue.'); return; }
    showResults(data);
  } catch (err) { endSearch(); showError('Impossible de contacter le serveur.'); }
});

function startSearch() {
  submitBtn.disabled = true;
  submitBtn.querySelector('.btn-text').textContent = 'Recherche…';
  loader.classList.remove('hidden');
  errorBox.classList.add('hidden');
  results.classList.add('hidden');
}
function endSearch() {
  submitBtn.disabled = false;
  submitBtn.querySelector('.btn-text').textContent = 'Estimer le prix';
  loader.classList.add('hidden');
}
function showError(msg) { errorMsg.textContent = msg; errorBox.classList.remove('hidden'); }

// ── Recap critères ──
function renderRecap() {
  const recap = document.getElementById('searchRecap');
  if (!recap) return;
  const yearMin = document.getElementById('year_min').value;
  const yearMax = document.getElementById('year_max').value;
  const kmMin   = document.getElementById('mileage_min').value;
  const kmMax   = document.getElementById('mileage_max').value;
  const fuel    = document.getElementById('fuel').value;
  const trans   = document.getElementById('transmission').value;
  const tags = [];
  if (brandSel.value) tags.push(`<strong>${brandSel.value}</strong> ${modelSel.value}`);
  if (yearMin && yearMax) tags.push(`${yearMin} — ${yearMax}`);
  else if (yearMin) tags.push(`Depuis ${yearMin}`);
  else if (yearMax) tags.push(`Jusqu'en ${yearMax}`);
  if (kmMin && kmMax) tags.push(`${Number(kmMin).toLocaleString('fr-FR')} — ${Number(kmMax).toLocaleString('fr-FR')} km`);
  else if (kmMin) tags.push(`Min. ${Number(kmMin).toLocaleString('fr-FR')} km`);
  else if (kmMax) tags.push(`Max. ${Number(kmMax).toLocaleString('fr-FR')} km`);
  if (fuel  && fuel  !== 'Tous')   tags.push(fuel);
  if (trans && trans !== 'Toutes') tags.push(trans);
  recap.innerHTML = tags.map(t => `<span class="recap-tag">${t}</span>`).join('');
}

// ── Résultats marché ──
function showResults(data) {
  renderRecap();
  document.getElementById('statMedian').textContent = fmt(data.median);
  document.getElementById('statRange').textContent  = `${fmt(data.range_low)} — ${fmt(data.range_high)}`;
  document.getElementById('statMean').textContent   = fmt(data.mean);
  document.getElementById('statMin').textContent    = fmt(data.min);
  document.getElementById('statMax').textContent    = fmt(data.max);
  document.getElementById('statStdev').textContent  = fmt(data.stdev);
  document.getElementById('statCount').textContent  =
    `${data.total_used} annonces analysées · ${data.total_raw} en base · outliers exclus`;

  // Bandeau si critères assouplis
  let noteEl = document.getElementById('sourceNote');
  if (!noteEl) {
    noteEl = document.createElement('div');
    noteEl.id = 'sourceNote';
    noteEl.className = 'source-note';
    document.getElementById('searchRecap').after(noteEl);
  }
  if (data.source_note) {
    noteEl.textContent = '⚠ ' + data.source_note;
    noteEl.style.display = 'block';
  } else {
    noteEl.style.display = 'none';
  }
  renderChart(data.histogram, data.median);
  showMLResults(data, data.median);
  renderListings(data.listings);
  results.classList.remove('hidden');
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function fmt(n) { return Number(n).toLocaleString('fr-FR') + ' €'; }

// ── Histogramme ──
function renderChart(histogram, median) {
  const ctx = document.getElementById('priceChart').getContext('2d');
  if (priceChart) priceChart.destroy();
  const labels = histogram.map(b => b.label);
  const counts = histogram.map(b => b.count);
  const maxCount = Math.max(...counts);
  priceChart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Annonces', data: counts,
      backgroundColor: counts.map(c => c === maxCount ? 'rgba(201,168,76,0.85)' : 'rgba(201,168,76,0.25)'),
      borderColor:     counts.map(c => c === maxCount ? 'rgba(201,168,76,1)'    : 'rgba(201,168,76,0.4)'),
      borderWidth: 1, borderRadius: 4 }] },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { backgroundColor: '#1e1e21', borderColor: '#2a2a2e', borderWidth: 1, titleColor: '#c9a84c', bodyColor: '#e8e4dc',
        callbacks: { title: i => `Tranche : ${i[0].label}`, label: i => `${i.raw} annonce(s)` } } },
      scales: { x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#7a7670', font: { size: 11 } } },
                y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#7a7670', font: { size: 11 }, stepSize: 1 } } } }
  });
}

// ── ML ──
async function checkMLStatus() {
  try {
    const res  = await fetch('/api/ml-status');
    const data = await res.json();
    const statusEl = document.getElementById('mlStatus');
    const textEl   = document.getElementById('mlStatusText');
    if (data.trained) {
      statusEl.className = 'ml-status ml-status--ready';
      const last = data.runs?.[0];
      const mae  = last?.mae ? ` · MAE ${Number(last.mae).toLocaleString('fr-FR')} €` : '';
      const r2   = last?.r2  ? ` · R² ${last.r2}` : '';
      textEl.textContent = `Modèle IA prêt${mae}${r2}`;
    } else {
      statusEl.className = 'ml-status ml-status--missing';
      textEl.textContent = 'Modèle IA non entraîné';
    }
  } catch (e) { document.getElementById('mlStatusText').textContent = 'Statut inconnu'; }
}

async function launchTraining() {
  const btn  = document.getElementById('btnTrain');
  const text = document.getElementById('mlStatusText');
  btn.disabled = true;
  document.getElementById('mlStatus').className = 'ml-status ml-status--training';
  text.textContent = 'Entraînement en cours… (2-5 min)';
  try {
    await fetch('/api/train', { method: 'POST' });
    const poll = setInterval(async () => {
      const res  = await fetch('/api/ml-status');
      const data = await res.json();
      if (data.trained && data.runs?.length) { clearInterval(poll); btn.disabled = false; checkMLStatus(); }
    }, 10000);
  } catch (e) { text.textContent = 'Erreur lors du lancement'; btn.disabled = false; }
}

function showMLResults(data, marketMedian) {
  const mlSection  = document.getElementById('mlSection');
  const predBlock  = document.getElementById('mlPredBlock');
  const notTrained = document.getElementById('mlNotTrained');
  const mlGrid     = predBlock ? predBlock.querySelector('.ml-grid') : null;

  // Toujours afficher la section IA
  mlSection.classList.remove('hidden');

  // Courbe de dépréciation — affichée dans tous les cas
  if (data.ml_depreciation && data.ml_depreciation.length > 0) {
    renderDepreciation(data.ml_depreciation, marketMedian, data.scatter_data);
  }

  // Bloc prédiction IA — seulement si modèle entraîné
  if (!data.ml_available || !data.ml_prediction) {
    if (mlGrid)      mlGrid.classList.add('hidden');
    if (notTrained)  notTrained.classList.remove('hidden');
    return;
  }
  if (mlGrid)      mlGrid.classList.remove('hidden');
  if (notTrained)  notTrained.classList.add('hidden');

  const pred = data.ml_prediction;
  document.getElementById('mlPredictedPrice').textContent = fmt(pred.predicted_price);
  document.getElementById('mlConfidence').textContent     = `Intervalle : ${fmt(pred.confidence_low)} — ${fmt(pred.confidence_high)}`;
  document.getElementById('mlMarketPrice').textContent    = fmt(marketMedian);
  document.getElementById('mlAiPrice').textContent        = fmt(pred.predicted_price);
  const diff   = pred.predicted_price - marketMedian;
  const diffEl = document.getElementById('mlDiff');
  diffEl.textContent = `${diff >= 0 ? '+' : ''}${Number(diff).toLocaleString('fr-FR')} €`;
  diffEl.className   = 'ml-compare-val ' + (diff >= 0 ? 'ml-diff--positive' : 'ml-diff--negative');
  renderDepreciation(data.ml_depreciation, pred.predicted_price, data.scatter_data);
}

function renderDepreciation(curve, currentPrice, scatterData) {
  const ctx = document.getElementById('deprecChart').getContext('2d');
  if (deprecChart) deprecChart.destroy();

  // ── Scatter : une annonce = un point (x=année, y=prix) ──
  const rawDots = (scatterData || []).filter(d => d.year && d.price);

  // Suppression des outliers visuels via IQR sur les prix
  const sortedPrices = rawDots.map(d => d.price).sort((a, b) => a - b);
  const q1 = sortedPrices[Math.floor(sortedPrices.length * 0.25)] || 0;
  const q3 = sortedPrices[Math.floor(sortedPrices.length * 0.75)] || Infinity;
  const iqr = q3 - q1;
  const priceLow  = q1 - 1.5 * iqr;
  const priceHigh = q3 + 1.5 * iqr;

  const dots = rawDots
    .filter(d => d.price >= priceLow && d.price <= priceHigh)
    .map(d => ({
      x: d.year,
      y: d.price,
      title:   d.title   || '',
      mileage: d.mileage || null,
      fuel:    d.fuel    || '',
    }));

  // ── Courbe de tendance (dépréciation) ──
  const trendLine = curve.map(p => ({ x: p.calendar_year, y: p.predicted_price }));

  // Calcul des bornes des axes (sans les outliers)
  const allYears  = [...dots.map(d => d.x), ...trendLine.map(t => t.x)];
  const allPrices = [...dots.map(d => d.y), ...trendLine.map(t => t.y)];
  const minYear   = Math.min(...allYears)  - 1;
  const maxYear   = Math.max(...allYears)  + 1;
  const maxPrice  = Math.max(...allPrices) * 1.15;

  deprecChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          // Points des annonces
          label: 'Annonces',
          data: dots,
          backgroundColor: 'rgba(255,255,255,0)',
          borderColor: 'rgba(201,168,76,0.6)',
          borderWidth: 1.5,
          pointRadius: 6,
          pointHoverRadius: 8,
          pointStyle: 'circle',
          order: 2,
        },
        {
          // Courbe de tendance / dépréciation
          label: 'Tendance',
          data: trendLine,
          type: 'line',
          borderColor: 'rgba(100,160,230,0.85)',
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
          tension: 0.4,
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1e1e21',
          borderColor: '#2a2a2e',
          borderWidth: 1,
          titleColor: '#c9a84c',
          bodyColor: '#e8e4dc',
          callbacks: {
            title: items => {
              const d = items[0].raw;
              return d.title ? d.title.slice(0, 45) : `Année ${d.x}`;
            },
            label: item => {
              const d = item.raw;
              const lines = [`Prix : ${Number(d.y).toLocaleString('fr-FR')} €`];
              if (d.mileage) lines.push(`${Number(d.mileage).toLocaleString('fr-FR')} km`);
              if (d.fuel)    lines.push(d.fuel);
              return lines;
            },
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          min: minYear,
          max: maxYear,
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: {
            color: '#7a7670', font: { size: 11 },
            callback: v => Math.round(v),
            stepSize: 1,
          },
          title: { display: true, text: "Année d'immatriculation", color: '#7a7670', font: { size: 11 } },
        },
        y: {
          min: 0,
          max: maxPrice,
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: {
            color: '#7a7670', font: { size: 11 },
            callback: v => `${Math.round(v / 1000)}k€`,
          },
          title: { display: true, text: 'Prix (€)', color: '#7a7670', font: { size: 11 } },
        },
      },
    },
  });

  // ── Table de dépréciation ──
  const tableEl = document.getElementById('deprecTable');
  tableEl.innerHTML = '';
  curve.forEach(p => {
    const cls  = p.pct_remaining >= 70 ? 'depre-pct--high' : p.pct_remaining >= 40 ? 'depre-pct--mid' : 'depre-pct--low';
    const cell = document.createElement('div');
    cell.className = 'depre-cell';
    cell.innerHTML = `
      <div class="depre-year">${p.calendar_year}</div>
      <div class="depre-price">${Math.round(p.predicted_price / 1000 * 10) / 10}k€</div>
      <div class="depre-pct ${cls}">${p.pct_remaining}%</div>`;
    tableEl.appendChild(cell);
  });
}

// ── Annonces ──
function renderListings(listings) {
  const grid  = document.getElementById('listingsGrid');
  const badge = document.getElementById('listingsBadge');
  grid.innerHTML = '';
  badge.textContent = `${listings.length} annonce(s)`;
  listings.forEach(lst => {
    const card = document.createElement(lst.url ? 'a' : 'div');
    card.className = 'listing-card';
    if (lst.url) { card.href = lst.url; card.target = '_blank'; card.rel = 'noopener'; }
    const tags = [];
    if (lst.year)         tags.push(`📅 ${lst.year}`);
    if (lst.mileage)      tags.push(`🛣 ${Number(lst.mileage).toLocaleString('fr-FR')} km`);
    if (lst.fuel)         tags.push(`⛽ ${lst.fuel}`);
    if (lst.transmission) tags.push(`⚙️ ${lst.transmission}`);
    if (lst.location)     tags.push(`📍 ${lst.location}`);
    card.innerHTML = `
      <div class="listing-title">${escHtml(lst.title || 'Annonce sans titre')}</div>
      <div class="listing-price">${Number(lst.price).toLocaleString('fr-FR')} €</div>
      <div class="listing-meta">${tags.map(t => `<span class="listing-tag">${t}</span>`).join('')}</div>
      <div class="listing-source">${lst.source}</div>`;
    grid.appendChild(card);
  });
}

function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

checkMLStatus();