# REST & WebSocket API Specification

## 1. Overview & Base URL

The **Urban Acoustic Intelligence Platform Backend** is powered by FastAPI and Uvicorn. It exposes high-throughput asynchronous REST endpoints and a real-time bidirectional WebSocket event hub.

- **Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`
- **WebSocket Live Stream**: `ws://127.0.0.1:8000/ws/live`

All standard REST endpoints accept and return UTF-8 encoded `application/json`. Timestamps use ISO-8601 formatting with UTC offset (e.g. `2026-10-15T00:00:00Z`).

---

## 2. Health & System Status

### `GET /health`
Returns the operational health status of the backend services, database connectivity, and streaming broker.

#### Response (200 OK)
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected (SQLITE)",
  "streaming_broker": "connected (LOCAL_QUEUE)",
  "timestamp": "2026-10-15T00:20:00.000000Z"
}
```

---

## 3. Acoustic Observations API

### `POST /observations`
Ingests a single acoustic observation event from a mobile or stationary sensor. Persists the record to the spatial database, attaches anomaly scoring, and asynchronously broadcasts the event to all connected dashboard WebSocket clients.

#### Request Body Schema
```json
{
  "event_id": "string (UUIDv4, required)",
  "device_id": "string (required, e.g. SENSOR-MOBILE-01)",
  "device_model": "string (optional, e.g. Apple iPhone 14 Pro)",
  "timestamp": "string (ISO-8601, required)",
  "latitude": "number (float between -90 and 90, required)",
  "longitude": "number (float between -180 and 180, required)",
  "gps_accuracy": "number (float in meters, optional)",
  "noise_level_db": "number (raw uncalibrated dB, required)",
  "sound_class": "string (UrbanSound8K class, required)",
  "confidence": "number (0.0 to 1.0, required)",
  "calibration_offset": "number (dB offset, optional, default: 0.0)",
  "calibrated_db": "number (noise_level_db + calibration_offset, optional)",
  "h3_cell": "string (H3 hexagonal index string, optional)",
  "anomaly_score": "number (z-score, optional)",
  "anomaly_severity": "string (NORMAL | ELEVATED | HIGH | CRITICAL, optional)",
  "anomaly_reasons": "array of strings (optional)"
}
```

#### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/observations" \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
    "device_id": "MOBILE-NODE-042",
    "device_model": "Apple iPhone 14 Pro",
    "timestamp": "2026-10-15T00:20:15Z",
    "latitude": 18.5204,
    "longitude": 73.8567,
    "gps_accuracy": 3.8,
    "noise_level_db": 75.8,
    "sound_class": "siren",
    "confidence": 0.94,
    "calibration_offset": 1.2,
    "calibrated_db": 77.0
  }'
```

#### Response (201 Created)
```json
{
  "id": 142,
  "event_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
  "device_id": "MOBILE-NODE-042",
  "device_model": "Apple iPhone 14 Pro",
  "timestamp": "2026-10-15T00:20:15Z",
  "latitude": 18.5204,
  "longitude": 73.8567,
  "gps_accuracy": 3.8,
  "noise_level_db": 75.8,
  "sound_class": "siren",
  "confidence": 0.94,
  "calibration_offset": 1.2,
  "calibrated_db": 77.0,
  "h3_cell": "8860885017fffff",
  "anomaly_score": 3.42,
  "anomaly_severity": "HIGH",
  "anomaly_reasons": ["Sound category 'siren' represents critical urban anomaly", "Decibel reading 77.0 dB exceeds zone limit 65.0 dB by 12.0 dB"],
  "created_at": "2026-10-15T00:20:16.123456Z"
}
```

---

### `GET /observations`
Queries observations supporting bounding box spatial filtering, time ranges, and sound category filters.

#### Query Parameters
- `min_lat` *(float, optional)*: Southern bounding box boundary.
- `min_lon` *(float, optional)*: Western bounding box boundary.
- `max_lat` *(float, optional)*: Northern bounding box boundary.
- `max_lon` *(float, optional)*: Eastern bounding box boundary.
- `start_time` *(string, optional)*: ISO-8601 start timestamp.
- `end_time` *(string, optional)*: ISO-8601 end timestamp.
- `sound_class` *(string, optional)*: Category filter (e.g. `siren`, `jackhammer`).
- `limit` *(int, optional, default: 100, max: 1000)*: Maximum records to return.

#### Response (200 OK)
```json
[
  {
    "id": 142,
    "event_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
    "device_id": "MOBILE-NODE-042",
    "timestamp": "2026-10-15T00:20:15Z",
    "latitude": 18.5204,
    "longitude": 73.8567,
    "noise_level_db": 75.8,
    "sound_class": "siren",
    "confidence": 0.94,
    "calibrated_db": 77.0,
    "h3_cell": "8860885017fffff"
  }
]
```

---

### `GET /observations/latest`
Retrieves the most recent $N$ observation records sorted chronologically descending.

#### Query Parameters
- `limit` *(int, optional, default: 50, min: 1, max: 500)*: Record limit.

---

## 4. Spatial Intelligence & Aggregation API

### `GET /spatial/heatmap`
Returns simplified coordinate-intensity tuples optimized for web mapping heatmap renderers (`Leaflet.heat` or `Deck.gl`).

#### Query Parameters
- `limit` *(int, optional, default: 500)*: Maximum points.

#### Response (200 OK)
```json
[
  [18.5204, 73.8567, 77.0],
  [18.5245, 73.8512, 64.5],
  [18.5310, 73.8601, 82.3]
]
```
Format: `[latitude, longitude, intensity_db]`.

---

### `GET /spatial/h3/grid`
Aggregates recent observations into discrete Uber H3 hexagonal cells with calculated energy-equivalent $L_{eq}$, peak decibels, dominant sound class, and observation counts.

#### Query Parameters
- `resolution` *(int, optional, default: 8, range: 6-10)*: Target H3 indexing resolution.
- `limit` *(int, optional, default: 500)*: Number of recent observations to aggregate.

#### Response (200 OK)
```json
[
  {
    "h3_cell": "8860885017fffff",
    "resolution": 8,
    "center_lat": 18.517551,
    "center_lon": 73.855273,
    "observation_count": 14,
    "average_db": 71.4,
    "energy_average_db": 74.8,
    "peak_db": 84.2,
    "min_db": 62.1,
    "dominant_sound_class": "car_horn",
    "sound_class_distribution": {
      "car_horn": 8,
      "engine_idling": 4,
      "siren": 2
    },
    "anomaly_count": 3
  }
]
```

---

### `GET /spatial/h3/grid/geojson`
Returns the H3 hexagonal grid as a fully compliant **GeoJSON FeatureCollection (RFC 7946)** ready for direct consumption by Leaflet, MapLibre, Mapbox, or QGIS.

#### Response (200 OK)
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [73.8512, 18.5140],
            [73.8561, 18.5122],
            [73.8605, 18.5158],
            [73.8599, 18.5211],
            [73.8550, 18.5229],
            [73.8506, 18.5193],
            [73.8512, 18.5140]
          ]
        ]
      },
      "properties": {
        "h3_cell": "8860885017fffff",
        "resolution": 8,
        "observation_count": 14,
        "average_db": 71.4,
        "energy_average_db": 74.8,
        "dominant_sound_class": "car_horn",
        "anomaly_count": 3
      }
    }
  ]
}
```

---

### `GET /spatial/anomalies`
Returns observation events flagged as statistical acoustic anomalies ($z \ge 2.0$ or severe class exceedance).

#### Query Parameters
- `min_score` *(float, optional, default: 2.0)*: Minimum anomaly score.
- `limit` *(int, optional, default: 50)*: Maximum anomaly records.

---

## 5. Analytics & Taxonomy API

### `GET /statistics`
Returns global platform metrics for dashboard summary cards.

#### Response (200 OK)
```json
{
  "total_observations": 1250,
  "sound_class_distribution": {
    "engine_idling": 340,
    "car_horn": 280,
    "siren": 85,
    "street_music": 190,
    "dog_bark": 120,
    "drilling": 95,
    "jackhammer": 80,
    "children_playing": 60
  },
  "baseline_mean_db": 66.8,
  "baseline_std_db": 6.4,
  "active_devices_count": 8
}
```

---

### `GET /sound-classes`
Returns the official UrbanSound8K taxonomy classes supported by the deep learning classifier.

#### Response (200 OK)
```json
{
  "classes": [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
  ],
  "total_classes": 10
}
```

---

## 6. Real-Time WebSocket Streaming Protocol

### `WebSocket /ws/live`
Clients open a persistent bidirectional WebSocket connection to receive real-time observation events and anomaly alerts as they are streamed into the backend.

- **URL**: `ws://127.0.0.1:8000/ws/live`
- **Protocol**: Text frames containing serialized JSON objects.

#### Connection Handshake
1. Browser client initiates standard WebSocket upgrade request to `/ws/live`.
2. Server registers connection in the global `ws_manager` hub.
3. Keep-alive: Clients may optionally send a JSON text frame `{"type": "ping"}`; the server responds immediately with `{"type": "pong"}`.

#### Downstream Broadcast Frame
Whenever an observation is ingested via REST or Kafka stream consumer, the server broadcasts the event:
```json
{
  "id": 142,
  "event_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
  "device_id": "MOBILE-NODE-042",
  "device_model": "Apple iPhone 14 Pro",
  "timestamp": "2026-10-15T00:20:15Z",
  "latitude": 18.5204,
  "longitude": 73.8567,
  "noise_level_db": 75.8,
  "sound_class": "siren",
  "confidence": 0.94,
  "calibration_offset": 1.2,
  "calibrated_db": 77.0,
  "h3_cell": "8860885017fffff",
  "anomaly_score": 3.42,
  "anomaly_severity": "HIGH",
  "anomaly_reasons": ["Sound category 'siren' represents critical urban anomaly"]
}
```

---

## 7. Error Handling & HTTP Status Codes

Errors return standard JSON payloads with descriptive details:
```json
{
  "detail": "Failed to record observation: UNIQUE constraint failed: noise_observations.event_id"
}
```

| HTTP Code | Name | Scenario |
|---|---|---|
| `200 OK` | Success | Successful GET request |
| `201 Created` | Created | Successful POST observation creation |
| `400 Bad Request` | Validation Error | Duplicate `event_id`, invalid coordinates, or malformed JSON |
| `404 Not Found` | Not Found | Route does not exist |
| `422 Unprocessable Entity` | Schema Error | Pydantic validation failure (e.g. missing required field) |
| `500 Internal Error` | Server Error | Unhandled server exception |
