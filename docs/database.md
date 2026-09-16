# Database Design

## Overview

The project uses PostgreSQL with PostGIS extension for spatial-temporal data storage.

## Schema

*Schema details will be added in Phase 6.*

## Tables

### noise_observations (planned)

| Column             | Type              | Description                           |
|--------------------|-------------------|---------------------------------------|
| id                 | SERIAL PRIMARY KEY| Auto-incrementing ID                  |
| event_id           | UUID              | Unique event identifier               |
| device_id          | VARCHAR(50)       | Sensor/device identifier              |
| timestamp          | TIMESTAMPTZ       | Observation timestamp                 |
| latitude           | DOUBLE PRECISION  | GPS latitude                          |
| longitude          | DOUBLE PRECISION  | GPS longitude                         |
| location           | GEOGRAPHY(POINT)  | PostGIS geographic point              |
| gps_accuracy       | FLOAT             | GPS accuracy in meters                |
| noise_level_db     | FLOAT             | Raw noise level (dB)                  |
| sound_class        | VARCHAR(50)       | Predicted sound category              |
| confidence         | FLOAT             | ML prediction confidence              |
| calibration_offset | FLOAT             | Device calibration offset (dB)        |
| calibrated_db      | FLOAT             | Calibrated noise level (dB)           |
| anomaly_score      | FLOAT             | Anomaly detection score               |
| h3_cell            | VARCHAR(20)       | H3 hexagonal cell index               |
| created_at         | TIMESTAMPTZ       | Record creation timestamp             |

---

*This document will be updated when the database is implemented.*
