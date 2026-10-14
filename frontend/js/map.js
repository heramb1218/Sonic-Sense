/**
 * Urban Acoustic Intelligence — Leaflet Map & Spatial Layer Controller
 */

export class AcousticMapManager {
  constructor(mapElementId = 'map-viewport') {
    this.mapElementId = mapElementId;
    this.map = null;
    this.heatmapLayer = null;
    this.hexagonLayer = null;
    this.sensorLayerGroup = null;
    this.sensorsMap = new Map(); // device_id -> L.marker
    this.defaultCenter = [18.5204, 73.8567]; // Default city center (or dynamic from points)
    this.defaultZoom = 13;

    this.initMap();
  }

  initMap() {
    // 1. Initialize Leaflet Map
    this.map = L.map(this.mapElementId, {
      center: this.defaultCenter,
      zoom: this.defaultZoom,
      zoomControl: false,
      attributionControl: false,
    });

    // Zoom control at bottom right
    L.control.zoom({ position: 'bottomright' }).addTo(this.map);

    // 2. Add CartoDB Dark Matter Base Tiles
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      subdomains: 'abcd',
      maxZoom: 19,
    }).addTo(this.map);

    // 3. Initialize Layer Groups
    this.sensorLayerGroup = L.layerGroup().addTo(this.map);
  }

  /**
   * Update continuous decibel heatmap layer.
   * @param {Array} points Array of {latitude, longitude, intensity}
   */
  updateHeatmap(points) {
    if (!points || !points.length) return;

    const heatData = points.map(p => [p.latitude, p.longitude, p.intensity]);

    if (this.heatmapLayer) {
      this.heatmapLayer.setLatLngs(heatData);
    } else if (typeof L.heatLayer === 'function') {
      this.heatmapLayer = L.heatLayer(heatData, {
        radius: 28,
        blur: 20,
        maxZoom: 17,
        max: 1.0,
        gradient: {
          0.1: '#10b981', // quiet
          0.35: '#06b6d4', // moderate
          0.6: '#f59e0b', // elevated
          0.8: '#f97316', // loud
          1.0: '#ef4444', // hazardous
        },
      }).addTo(this.map);
    }
  }

  /**
   * Determine color corresponding to calibrated decibels.
   */
  getDecibelColor(db) {
    if (db < 55.0) return '#10b981'; // Quiet Emerald
    if (db < 65.0) return '#06b6d4'; // Moderate Cyan
    if (db < 75.0) return '#f59e0b'; // Elevated Amber
    if (db < 85.0) return '#f97316'; // Loud Orange
    return '#ef4444';                // Hazardous Crimson
  }

  /**
   * Render Uber H3 Hexagonal polygon aggregation layer.
   * @param {Object} geojsonFeatureCollection
   */
  updateHexagons(geojsonFeatureCollection) {
    if (!geojsonFeatureCollection || !geojsonFeatureCollection.features) return;

    if (this.hexagonLayer) {
      this.map.removeLayer(this.hexagonLayer);
    }

    this.hexagonLayer = L.geoJSON(geojsonFeatureCollection, {
      style: (feature) => {
        const avgDb = feature.properties.average_db || 60.0;
        const color = this.getDecibelColor(avgDb);
        return {
          fillColor: color,
          fillOpacity: 0.55,
          color: 'rgba(255, 255, 255, 0.25)',
          weight: 1.2,
        };
      },
      onEachFeature: (feature, layer) => {
        const props = feature.properties;
        const soundClass = props.dominant_sound_class || 'ambient';
        const color = this.getDecibelColor(props.average_db);

        // Hover effect
        layer.on({
          mouseover: (e) => {
            const target = e.target;
            target.setStyle({
              fillOpacity: 0.85,
              weight: 2.5,
              color: '#ffffff',
            });
            target.bringToFront();
          },
          mouseout: (e) => {
            this.hexagonLayer.resetStyle(e.target);
          },
        });

        // Popup content
        const popupHtml = `
          <div class="popup-custom">
            <h4>H3 Hexagon Cell</h4>
            <div class="popup-row"><span>Cell ID:</span><span>${props.h3_cell}</span></div>
            <div class="popup-row"><span>Average Noise:</span><span style="color:${color}">${props.average_db} dB</span></div>
            <div class="popup-row"><span>Peak Noise:</span><span>${props.peak_db} dB</span></div>
            <div class="popup-row"><span>Dominant Class:</span><span>${soundClass}</span></div>
            <div class="popup-row"><span>Observations:</span><span>${props.observation_count}</span></div>
            <div class="popup-row"><span>Anomalies:</span><span style="color:${props.anomaly_count > 0 ? '#ef4444' : '#10b981'}">${props.anomaly_count}</span></div>
          </div>
        `;
        layer.bindPopup(popupHtml);
      },
    }).addTo(this.map);
  }

  /**
   * Add or smoothly update a sensor marker on the map.
   */
  addOrUpdateSensor(observation) {
    const { device_id, latitude, longitude, calibrated_db, sound_class, confidence, device_model } = observation;
    if (!device_id || latitude == null || longitude == null) return;

    const latLng = [latitude, longitude];
    const markerColor = this.getDecibelColor(calibrated_db);

    const customIcon = L.divIcon({
      className: 'custom-sensor-div-icon',
      html: `
        <div style="
          width: 16px; 
          height: 16px; 
          border-radius: 50%; 
          background: ${markerColor}; 
          border: 2px solid #ffffff; 
          box-shadow: 0 0 10px ${markerColor};
          animation: pulse-animation 2s infinite;
        "></div>
      `,
      iconSize: [16, 16],
      iconAnchor: [8, 8],
    });

    if (this.sensorsMap.has(device_id)) {
      const marker = this.sensorsMap.get(device_id);
      marker.setLatLng(latLng);
      marker.setIcon(customIcon);
      marker.setPopupContent(this.createSensorPopup(observation));
    } else {
      const marker = L.marker(latLng, { icon: customIcon });
      marker.bindPopup(this.createSensorPopup(observation));
      this.sensorLayerGroup.addLayer(marker);
      this.sensorsMap.set(device_id, marker);
    }
  }

  createSensorPopup(obs) {
    const color = this.getDecibelColor(obs.calibrated_db);
    return `
      <div class="popup-custom">
        <h4>Sensor Node</h4>
        <div class="popup-row"><span>Device ID:</span><span>${obs.device_id}</span></div>
        <div class="popup-row"><span>Model:</span><span>${obs.device_model || 'IoT Probe'}</span></div>
        <div class="popup-row"><span>Calibrated Noise:</span><span style="color:${color}">${obs.calibrated_db} dB</span></div>
        <div class="popup-row"><span>Detected Sound:</span><span>${obs.sound_class} (${Math.round((obs.confidence || 0) * 100)}%)</span></div>
        <div class="popup-row"><span>Coordinates:</span><span>${obs.latitude.toFixed(4)}, ${obs.longitude.toFixed(4)}</span></div>
      </div>
    `;
  }

  /**
   * Pan and zoom directly to a spatial coordinate with animation.
   */
  flyTo(lat, lon, zoom = 15) {
    this.map.flyTo([lat, lon], zoom, { duration: 1.2 });
  }

  /**
   * Fit map view to bounds of all active sensors or hexagons.
   */
  fitToFeatures() {
    if (this.hexagonLayer && this.hexagonLayer.getLayers().length > 0) {
      this.map.fitBounds(this.hexagonLayer.getBounds(), { padding: [50, 50] });
    }
  }

  toggleHeatmap(visible) {
    if (!this.heatmapLayer) return;
    if (visible) {
      this.map.addLayer(this.heatmapLayer);
    } else {
      this.map.removeLayer(this.heatmapLayer);
    }
  }

  toggleHexagons(visible) {
    if (!this.hexagonLayer) return;
    if (visible) {
      this.map.addLayer(this.hexagonLayer);
    } else {
      this.map.removeLayer(this.hexagonLayer);
    }
  }

  toggleSensors(visible) {
    if (!this.sensorLayerGroup) return;
    if (visible) {
      this.map.addLayer(this.sensorLayerGroup);
    } else {
      this.map.removeLayer(this.sensorLayerGroup);
    }
  }
}
