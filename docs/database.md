# Database Architecture and Spatial-Temporal Design

## Overview

The Urban Noise Intelligence platform employs a spatial-temporal database design capable of running on **PostgreSQL with PostGIS** or a standalone **local SQLite database** (`urban_noise.db`) via an automated fallback engine.

---

## Table: `noise_observations`

Stores every acoustic observation emitted by the sensing network.

| Column | Type | Constraints | Description |
|:---|:---|:---|:---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique sequence identifier |
| `event_id` | VARCHAR(36) | UNIQUE, NOT NULL, INDEX | UUID for message deduplication |
| `device_id` | VARCHAR(50) | NOT NULL, INDEX | Originating sensing device ID |
| `device_model` | VARCHAR(100) | NULLABLE | Smartphone or hardware node model |
| `timestamp` | TIMESTAMPTZ | NOT NULL, INDEX | Instantaneous time of audio capture |
| `latitude` | FLOAT | NOT NULL | GPS latitude (WGS 84) |
| `longitude` | FLOAT | NOT NULL | GPS longitude (WGS 84) |
| `gps_accuracy` | FLOAT | NULLABLE | Simulated accuracy radius in meters |
| `noise_level_db`| FLOAT | NOT NULL | Raw sound pressure level estimate (dB SPL) |
| `sound_class` | VARCHAR(50) | NOT NULL, INDEX | ML predicted acoustic category |
| `confidence` | FLOAT | NOT NULL | Model prediction confidence score [0.0 - 1.0] |
| `calibration_offset` | FLOAT | NOT NULL | Hardware calibration offset in dB |
| `calibrated_db` | FLOAT | NOT NULL, INDEX | Calibrated noise level (`raw_db + offset`) |
| `h3_cell` | VARCHAR(20) | NULLABLE, INDEX | Uber H3 hexagonal spatial cell index |
| `anomaly_score` | FLOAT | NULLABLE | Anomaly z-score deviation |
| `audio_file` | VARCHAR(255) | NULLABLE | Associated audio file identifier |
| `created_at` | TIMESTAMPTZ | NOT NULL | Server ingestion timestamp |

---

## Spatial and Temporal Indexes

High-throughput queries rely on three dedicated multi-column compound indexes:
1. `idx_spatial_coords` ON `(latitude, longitude)`: Optimizes geographic bounding box and spatial radius queries.
2. `idx_temporal_class` ON `(timestamp, sound_class)`: Accelerates category-specific time-series charts.
3. `idx_h3_timestamp` ON `(h3_cell, timestamp)`: Powers fast hexagonal cell aggregations across time windows.

---

## Implemented Spatial-Temporal Queries (Section 20)

The `ObservationRepository` class exposes:
1. **`get_latest_observations(limit)`**: Retrieves most recent observations sorted chronologically.
2. **`get_observations_in_time_range(start_time, end_time)`**: Time-series window filtering.
3. **`get_observations_in_bounding_box(min_lat, min_lon, max_lat, max_lon)`**: Fast polygon/box viewport filtering.
4. **`get_observations_in_radius(center_lat, center_lon, radius_meters)`**: Haversine radial distance search.
5. **`get_observations_by_class(sound_class, limit)`**: Category filtering.
6. **`get_average_noise_by_area(min_lat, min_lon, max_lat, max_lon)`**: Area-level noise intensity aggregation (`avg`, `min`, `max`, `count`).
7. **`get_sound_event_counts()`**: Global breakdown of events per acoustic class.
8. **`get_historical_baseline(time_window_hours)`**: Baseline noise mean and variance calculation for anomaly detection.
9. **`get_anomalies(min_score)`**: Filters observations exceeding anomalous deviation threshold.
