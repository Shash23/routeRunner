/* RolledBadger — single-page app */
const API = ''; // same origin so cookies are sent
const fetchOpts = { credentials: 'include' };

let map = null;
let routeLayer = null;
let trainingInterval = null;
let polylineAnimationId = null;

const $ = (id) => document.getElementById(id);
const showState = (name) => {
  document.querySelectorAll('.state-only').forEach(el => el.classList.remove('active'));
  const el = document.querySelector(`[data-state="${name}"]`);
  if (el) el.classList.add('active');
};
const showTabs = (show) => {
  document.querySelector('.tabs-wrap').classList.toggle('hidden', !show);
};
const setTab = (tabName) => {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
  document.querySelectorAll('[data-tab]').forEach(el => {
    if (el.classList.contains('tab')) return;
    el.classList.toggle('hidden', el.dataset.tab !== tabName);
  });
};

// Decode Google polyline (simplified)
function decodePolyline(encoded) {
  if (!encoded) return [];
  const points = [];
  let i = 0, lat = 0, lon = 0;
  while (i < encoded.length) {
    let b, shift = 0, result = 0;
    do {
      b = encoded.charCodeAt(i++) - 63;
      result |= (b & 31) << shift;
      shift += 5;
    } while (b >= 32);
    const dlat = (result & 1) ? ~(result >> 1) : (result >> 1);
    shift = 0; result = 0;
    do {
      b = encoded.charCodeAt(i++) - 63;
      result |= (b & 31) << shift;
      shift += 5;
    } while (b >= 32);
    const dlon = (result & 1) ? ~(result >> 1) : (result >> 1);
    lat += dlat;
    lon += dlon;
    points.push([lat / 1e5, lon / 1e5]);
  }
  return points;
}

function drawMap(polylineEncoded, unavailable, unavailableMessage) {
  const wrap = $('map-wrap');
  if (!wrap) return;
  if (polylineAnimationId != null) {
    cancelAnimationFrame(polylineAnimationId);
    polylineAnimationId = null;
  }
  if (map) map.remove();
  map = null;
  routeLayer = null;
  if (unavailable) {
    const msg = unavailableMessage || 'Route map unavailable (service or network). Your workout and AI explanation are still valid.';
    wrap.innerHTML = `<div class="map-unavailable">${msg}</div>`;
    return;
  }
  wrap.innerHTML = '<div id="map"></div>';
  const L = window.L;
  if (!L) return;
  const centerLat = 43.0731, centerLon = -89.4012;
  map = L.map('map').setView([centerLat, centerLon], 14);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap'
  }).addTo(map);
  const points = decodePolyline(polylineEncoded);
  if (points.length >= 2) {
    const fullBounds = L.latLngBounds(points);
    map.fitBounds(fullBounds, { padding: [20, 20] });
    routeLayer = L.polyline([points[0], points[1]], { color: '#c5050c', weight: 4 }).addTo(map);
    const durationMs = 5000;
    const startTime = performance.now();
    function animate(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / durationMs);
      const endIndex = 1 + Math.floor(progress * (points.length - 1));
      const visiblePoints = points.slice(0, endIndex + 1);
      routeLayer.setLatLngs(visiblePoints);
      if (progress < 1) {
        polylineAnimationId = requestAnimationFrame(animate);
      } else {
        polylineAnimationId = null;
      }
    }
    polylineAnimationId = requestAnimationFrame(animate);
  }
}

async function api(path, options = {}) {
  const res = await fetch(`${API}${path}`, { ...fetchOpts, ...options });
  if (res.status === 401) return { _401: true };
  if (res.status === 503) return { _503: true };
  const data = await res.json().catch(() => ({}));
  if (res.status !== 200) return { _err: data.error || res.status };
  return data;
}

function checkAuth() {
  return api('/me').then(data => {
    if (data._401) return 'not_connected';
    if (data._503) return 'training';
    return 'ready';
  });
}

function startTrainingPoll() {
  showState('training');
  showTabs(false);
  const statusEl = $('training-status');
  const runsEl = $('training-runs');
  function poll() {
    api('/training/status').then(data => {
      if (data._401) {
        showState('not_connected');
        return;
      }
      if (data.state === 'failed') {
        statusEl.textContent = 'Training failed. Please try connecting again.';
        statusEl.classList.add('error');
        statusEl.classList.remove('loading');
        return;
      }
      if (data.state === 'ready') {
        if (trainingInterval) clearInterval(trainingInterval);
        loadRecommendation();
        return;
      }
      statusEl.textContent = 'Building your training profile…';
      statusEl.classList.add('loading');
      if (runsEl) runsEl.textContent = data.runs_loaded ?? 0;
    });
  }
  poll();
  trainingInterval = setInterval(poll, 3000);
}

function loadRecommendation() {
  showState('ready');
  showTabs(true);
  setTab('recommended');
  Promise.all([
    api('/recommendation/today'),
    api('/route?lat=43.0731&lon=-89.4012'),
    api('/recommendation/explain')
  ]).then(([rec, route, explain]) => {
    if (rec._401 || explain._401) {
      showState('not_connected');
      return;
    }
    if (rec._503 || explain._503) {
      startTrainingPoll();
      return;
    }
    if (rec._err || explain._err) return;
    // If route failed (500), still show recommendation + explanation; map will show "unavailable"
    const routeData = (route._err || route._503) ? { polyline: '', _unavailable: true } : route;
    if (routeData.error) routeData._unavailable = true;
    renderRecommendation(rec, routeData, explain);
  });
}

function renderRecommendation(rec, route, explain) {
  $('decision-workout').textContent = rec.workout_type || '—';
  $('decision-readiness').textContent = rec.readiness || '—';
  $('detail-distance').textContent = rec.distance_miles != null ? `${rec.distance_miles} mi` : '—';
  $('detail-pace').textContent = rec.pace_range || '—';
  $('detail-hr').textContent = rec.hr_zone || '—';
  const whyEl = $('detail-why');
  whyEl.innerHTML = '';
  (rec.why || []).forEach(w => {
    const li = document.createElement('li');
    li.textContent = w;
    whyEl.appendChild(li);
  });
  const distInput = $('input-distance');
  if (distInput && rec.distance_miles != null) distInput.value = rec.distance_miles;
  const hrSelect = $('input-hr');
  if (hrSelect && rec.hr_zone) hrSelect.value = rec.hr_zone;
  drawMap(route.polyline, route._unavailable, route.error);
  renderExplain(explain);
  $('adjust-msg-wrap').classList.add('hidden');
}

function renderExplain(explain) {
  if (explain._err || explain._401 || explain._503) return;
  $('explain-decision').textContent = explain.decision || '—';
  $('explain-fatigue').textContent = explain.fatigue_score != null ? explain.fatigue_score : '—';
  $('explain-strength').textContent = explain.personalization_strength != null ? explain.personalization_strength : '—';
  $('explain-model').textContent = explain.model_type || '—';
}

function onAdjust() {
  const distance = parseFloat($('input-distance').value);
  const hrZone = $('input-hr').value;
  const body = {};
  if (!isNaN(distance) && distance > 0) body.distance_miles = distance;
  if (hrZone) body.hr_zone = hrZone;
  $('adjust-msg-wrap').classList.add('hidden');
  api('/recommendation/adjust', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  }).then(data => {
    if (data._401) { showState('not_connected'); return; }
    if (data._503) { startTrainingPoll(); return; }
    if (data._err) return;
    const msgEl = $('adjust-msg');
    msgEl.textContent = data.message || 'Updated.';
    msgEl.classList.toggle('error', !!data._err);
    $('adjust-msg-wrap').classList.remove('hidden');
    $('detail-distance').textContent = data.distance_miles != null ? `${data.distance_miles} mi` : '—';
    $('decision-workout').textContent = data.workout_type || $('decision-workout').textContent;
    drawMap(data.polyline);
    return api('/recommendation/explain').then(renderExplain);
  });
}

function init() {
  $('btn-connect').addEventListener('click', (e) => {
    e.preventDefault();
    window.location.href = (API || window.location.origin) + '/auth/strava';
  });

  document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', () => setTab(t.dataset.tab));
  });

  const distEl = $('input-distance');
  const hrEl = $('input-hr');
  if (distEl) distEl.addEventListener('change', onAdjust);
  if (hrEl) hrEl.addEventListener('change', onAdjust);

  checkAuth().then(state => {
    if (state === 'not_connected') {
      showState('not_connected');
      showTabs(false);
      return;
    }
    if (state === 'training') {
      startTrainingPoll();
      return;
    }
    loadRecommendation();
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
