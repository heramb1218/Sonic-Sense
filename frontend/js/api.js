/**
 * Urban Acoustic Intelligence — REST API Client Module
 */

const API_BASE = window.location.origin;

export const apiClient = {
  /**
   * Check backend service health and environment status.
   */
  async getHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.warn('[API] Health check failed:', err);
      return null;
    }
  },

  /**
   * Fetch global platform statistics (observations, distributions, baseline).
   */
  async getStatistics() {
    try {
      const res = await fetch(`${API_BASE}/statistics`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.warn('[API] Statistics fetch failed:', err);
      return null;
    }
  },

  /**
   * Fetch UrbanSound8K taxonomy classes catalog.
   */
  async getSoundClasses() {
    try {
      const res = await fetch(`${API_BASE}/sound-classes`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.warn('[API] Sound classes fetch failed:', err);
      return null;
    }
  },

  /**
   * Fetch spatial points for heatmap rendering.
   */
  async getHeatmapPoints(limit = 300) {
    try {
      const res = await fetch(`${API_BASE}/heatmap?limit=${limit}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.warn('[API] Heatmap fetch failed:', err);
      return [];
    }
  },

  /**
   * Fetch Uber H3 hexagonal aggregation layer as GeoJSON.
   */
  async getHexagons(resolution = 8, limit = 500) {
    try {
      const res = await fetch(`${API_BASE}/hexagons?resolution=${resolution}&limit=${limit}&geojson=true`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.warn('[API] Hexagons fetch failed:', err);
      return null;
    }
  },

  /**
   * Fetch latest acoustic observations.
   */
  async getLatestObservations(limit = 50) {
    try {
      const res = await fetch(`${API_BASE}/observations/latest?limit=${limit}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.warn('[API] Latest observations fetch failed:', err);
      return [];
    }
  },

  /**
   * Fetch acoustic anomaly events.
   */
  async getAnomalies(minScore = 2.0, limit = 50) {
    try {
      const res = await fetch(`${API_BASE}/anomalies?min_score=${minScore}&limit=${limit}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.warn('[API] Anomalies fetch failed:', err);
      return [];
    }
  },

  /**
   * Ingest a synthetic event for interactive dashboard demonstrations.
   */
  async postObservation(eventPayload) {
    try {
      const res = await fetch(`${API_BASE}/observations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(eventPayload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.error('[API] Ingest observation failed:', err);
      return null;
    }
  },
};
