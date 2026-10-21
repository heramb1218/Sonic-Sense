/**
 * Urban Acoustic Intelligence — Main Application Controller
 */

import { apiClient } from './api.js';
import { AcousticMapManager } from './map.js?v=2';
import { AcousticChartsManager } from './charts.js';
import { AcousticWebSocketClient } from './websocket.js';

class AcousticDashboardApp {
  constructor() {
    this.mapManager = null;
    this.chartsManager = null;
    this.wsClient = null;

    this.activeSensorsCount = 0;
    this.anomalyCount = 0;
    this.soundClassCounts = {};
    this.recentObservations = [];

    this.init();
  }

  async init() {
    console.log('[App] Initializing Urban Acoustic Intelligence Dashboard...');

    // 1. Initialize Visual Components
    this.mapManager = new AcousticMapManager('map-viewport');
    this.chartsManager = new AcousticChartsManager('sound-class-chart');

    // 2. Setup Event Listeners for UI Controls
    this.setupUIListeners();

    // 3. Load Initial Platform Data via REST API
    await this.loadInitialData();

    // 4. Connect Live WebSocket Stream
    this.wsClient = new AcousticWebSocketClient('/ws/live', (event) => this.handleLiveEvent(event));

    // 5. Periodic Background Refresh for Aggregates (every 30 seconds)
    setInterval(() => this.refreshAggregates(), 30000);
  }

  async loadInitialData() {
    // Health & System Dialect
    const health = await apiClient.getHealth();
    if (health) {
      const dbBadge = document.getElementById('db-status-badge');
      if (dbBadge) dbBadge.textContent = `DB: ${health.database_engine}`;
    }

    // Global Statistics & KPI Cards
    const stats = await apiClient.getStatistics();
    if (stats) {
      this.soundClassCounts = stats.sound_class_distribution || {};
      this.chartsManager.updateClassDistribution(this.soundClassCounts);

      const avgDbEl = document.getElementById('kpi-avg-db');
      if (avgDbEl) avgDbEl.textContent = stats.baseline_mean_db ? stats.baseline_mean_db.toFixed(1) : '58.4';

      const sensorCountEl = document.getElementById('kpi-active-sensors');
      if (sensorCountEl) sensorCountEl.textContent = stats.active_devices_count || '15';
    }

    // Heatmap Layer
    const heatmapPoints = await apiClient.getHeatmapPoints(400);
    if (heatmapPoints.length > 0) {
      this.mapManager.updateHeatmap(heatmapPoints);
    }

    // Uber H3 Hexagons Layer
    const hexagonGeoJSON = await apiClient.getHexagons(8, 500);
    if (hexagonGeoJSON) {
      this.mapManager.updateHexagons(hexagonGeoJSON);
      this.mapManager.fitToFeatures();
    }

    // Latest Observations & Sensors
    const latestObs = await apiClient.getLatestObservations(50);
    if (latestObs.length > 0) {
      latestObs.forEach((obs) => this.mapManager.addOrUpdateSensor(obs));
      const mostRecent = latestObs[0];
      AcousticChartsManager.updateDecibelMeter(mostRecent.calibrated_db);

      const peakDb = Math.max(...latestObs.map((o) => o.calibrated_db));
      const peakEl = document.getElementById('kpi-peak-db');
      if (peakEl) peakEl.textContent = peakDb.toFixed(1);
    }

    // Anomalies
    const anomalies = await apiClient.getAnomalies(2.0, 30);
    this.anomalyCount = anomalies.length;
    const anomCountEl = document.getElementById('kpi-anomalies-count');
    if (anomCountEl) anomCountEl.textContent = this.anomalyCount;

    const feedContainer = document.getElementById('anomaly-feed-list');
    if (feedContainer) {
      feedContainer.innerHTML = '';
      anomalies.forEach((anom) => this.renderAnomalyCard(anom, false));
    }
  }

  handleLiveEvent(event) {
    // 1. Update Decibel VU Meter
    const db = event.calibrated_db || event.noise_level_db || 60.0;
    AcousticChartsManager.updateDecibelMeter(db);

    // 2. Update Map Sensor Marker
    this.mapManager.addOrUpdateSensor(event);

    // 3. Update Sound Class Distribution Count
    const soundClass = event.sound_class;
    if (soundClass) {
      this.soundClassCounts[soundClass] = (this.soundClassCounts[soundClass] || 0) + 1;
      this.chartsManager.updateClassDistribution(this.soundClassCounts);
    }

    // 4. Update Peak Decibel KPI if new peak
    const peakEl = document.getElementById('kpi-peak-db');
    if (peakEl) {
      const currentPeak = parseFloat(peakEl.textContent) || 0;
      if (db > currentPeak) {
        peakEl.textContent = db.toFixed(1);
      }
    }

    // 5. Handle Anomaly Alert
    const anomalyScore = event.anomaly_score || 0;
    const severity = event.anomaly_severity || (anomalyScore >= 3.0 ? 'CRITICAL' : anomalyScore >= 2.0 ? 'HIGH' : 'NORMAL');

    if (severity !== 'NORMAL' || anomalyScore >= 2.0) {
      this.anomalyCount++;
      const anomCountEl = document.getElementById('kpi-anomalies-count');
      if (anomCountEl) anomCountEl.textContent = this.anomalyCount;

      this.renderAnomalyCard({
        event_id: event.event_id,
        timestamp: event.timestamp || new Date().toISOString(),
        latitude: event.latitude,
        longitude: event.longitude,
        sound_class: event.sound_class,
        calibrated_db: db,
        anomaly_score: anomalyScore,
        severity: severity,
        reasons: event.anomaly_reasons || [`Elevated acoustic spike (${db} dB)`],
      }, true);
    }
  }

  renderAnomalyCard(anom, prepend = true) {
    const feed = document.getElementById('anomaly-feed-list');
    if (!feed) return;

    const severity = anom.severity || (anom.anomaly_score >= 3.0 ? 'CRITICAL' : 'MODERATE');
    const badgeClass = severity === 'CRITICAL' ? 'badge-critical' : severity === 'HIGH' ? 'badge-high' : 'badge-moderate';
    const borderColor = severity === 'CRITICAL' ? '#ef4444' : severity === 'HIGH' ? '#f97316' : '#f59e0b';

    const card = document.createElement('div');
    card.className = 'anomaly-feed-item';
    card.style.setProperty('--item-border', borderColor);

    const timeStr = new Date(anom.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const reasons = (anom.reasons && anom.reasons.length) ? anom.reasons[0] : `Acoustic deviation (Score: ${anom.anomaly_score})`;

    card.innerHTML = `
      <div class="anomaly-item-header">
        <span class="badge ${badgeClass}">${severity}</span>
        <span class="anomaly-item-time">${timeStr}</span>
      </div>
      <div class="anomaly-item-body">
        <strong>${(anom.sound_class || 'Sound Event').toUpperCase()}</strong>
        <span style="font-family: var(--font-mono); font-weight:600; color: ${borderColor}">${anom.calibrated_db.toFixed(1)} dB</span>
      </div>
      <div class="anomaly-item-reason">${reasons}</div>
    `;

    // Click on alert zooms and centers map on coordinates
    if (anom.latitude && anom.longitude) {
      card.addEventListener('click', () => {
        this.mapManager.flyTo(anom.latitude, anom.longitude, 16);
      });
    }

    if (prepend && feed.firstChild) {
      feed.insertBefore(card, feed.firstChild);
    } else {
      feed.appendChild(card);
    }

    // Keep feed trimmed to 50 cards
    while (feed.children.length > 50) {
      feed.removeChild(feed.lastChild);
    }
  }

  async refreshAggregates() {
    const hexagons = await apiClient.getHexagons(8, 500);
    if (hexagons) this.mapManager.updateHexagons(hexagons);

    const heatmap = await apiClient.getHeatmapPoints(300);
    if (heatmap.length) this.mapManager.updateHeatmap(heatmap);
  }

  setupUIListeners() {
    // Layer Toggles
    const toggleHeatmap = document.getElementById('toggle-heatmap');
    if (toggleHeatmap) {
      toggleHeatmap.addEventListener('change', (e) => this.mapManager.toggleHeatmap(e.target.checked));
    }

    const toggleHexagons = document.getElementById('toggle-hexagons');
    if (toggleHexagons) {
      toggleHexagons.addEventListener('change', (e) => this.mapManager.toggleHexagons(e.target.checked));
    }

    const toggleSensors = document.getElementById('toggle-sensors');
    if (toggleSensors) {
      toggleSensors.addEventListener('change', (e) => this.mapManager.toggleSensors(e.target.checked));
    }

    // Reset View Button
    const resetBtn = document.getElementById('btn-reset-view');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => this.mapManager.fitToFeatures());
    }

    // Simulate Event Button
    const simulateBtn = document.getElementById('btn-simulate-event');
    if (simulateBtn) {
      simulateBtn.addEventListener('click', () => this.triggerSyntheticEvent());
    }
  }

  async triggerSyntheticEvent() {
    // Generate an illustrative event near map center
    const isAnomaly = Math.random() < 0.4;
    const baseLat = 18.5204 + (Math.random() - 0.5) * 0.03;
    const baseLon = 73.8567 + (Math.random() - 0.5) * 0.03;

    const classes = ['siren', 'drilling', 'street_music', 'car_horn', 'engine_idling', 'dog_bark'];
    const chosenClass = isAnomaly ? 'siren' : classes[Math.floor(Math.random() * classes.length)];
    const db = isAnomaly ? 88.0 + Math.random() * 8.0 : 58.0 + Math.random() * 15.0;

    const payload = {
      event_id: `SIM-CLICK-${Date.now()}`,
      device_id: `SENSOR-PROBE-${Math.floor(Math.random() * 8 + 1)}`,
      device_model: 'Crowdsourced-SmartPhone',
      timestamp: new Date().toISOString(),
      latitude: parseFloat(baseLat.toFixed(6)),
      longitude: parseFloat(baseLon.toFixed(6)),
      noise_level_db: parseFloat(db.toFixed(1)),
      sound_class: chosenClass,
      confidence: parseFloat((0.85 + Math.random() * 0.12).toFixed(2)),
      calibration_offset: -1.5,
      calibrated_db: parseFloat((db - 1.5).toFixed(1)),
    };

    console.log('[App] Sending simulated event:', payload);
    await apiClient.postObservation(payload);
  }
}

// Launch application on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.acousticApp = new AcousticDashboardApp();
});
