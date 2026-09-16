# System Architecture

## Overview

The Urban Noise Intelligence platform follows a pipeline architecture:

```
Audio Capture → Feature Extraction → ML Classification → Streaming → Storage → Aggregation → Anomaly Detection → API → Dashboard
```

## Detailed Architecture

```
         URBAN ENVIRONMENT
                ↓
    SIMULATED SMARTPHONE SENSORS
                ↓
         AUDIO PROCESSING (Librosa)
                ↓
         LOG-MEL FEATURES
                ↓
         PYTORCH CNN MODEL
                ↓
       SOUND CLASS + CONFIDENCE
                ↓
      GPS + TIMESTAMP + METADATA
                ↓
             KAFKA
                ↓
      PROCESSING / CONSUMER
          ↙           ↘
   ANOMALY ENGINE     DATABASE
          ↓          PostgreSQL
                     + PostGIS
                        ↓
                       H3
                        ↓
                    FASTAPI
                   ↙        ↘
            REST API       WebSocket
                ↓             ↓
                └──────┬──────┘
                       ↓
                WEB DASHBOARD
                       ↓
          REAL-TIME URBAN NOISE MAP
```

## Components

### 1. Audio Processing Layer
- **Library**: Librosa
- **Input**: Raw `.wav` files
- **Output**: Log-Mel Spectrograms (128 mel bins, 22050 Hz sample rate)
- **Purpose**: Converts raw audio into a visual representation suitable for CNN processing

### 2. ML Classification Layer
- **Framework**: PyTorch
- **Model**: CNN (baseline), optional Transformer (comparison)
- **Input**: Log-Mel Spectrograms
- **Output**: Sound class label + confidence score
- **Classes**: 10 urban sound categories from UrbanSound8K

### 3. Sensor Simulation Layer
- **Purpose**: Simulates smartphones/acoustic sensors in an urban environment
- **Generates**: GPS coordinates, timestamps, device metadata, noise levels
- **Output**: Structured noise observation events

### 4. Streaming Layer
- **Technology**: Apache Kafka
- **Topic**: `urban-noise-events`
- **Purpose**: Decouples sensor data production from consumption, enables real-time flow

### 5. Storage Layer
- **Database**: PostgreSQL + PostGIS
- **Purpose**: Stores spatial-temporal observations with geographic indexing
- **Features**: Spatial queries, time-range queries, geographic filtering

### 6. Spatial Aggregation Layer
- **Technology**: H3 (Uber's hexagonal spatial indexing)
- **Purpose**: Aggregates observations into hexagonal cells for summary statistics
- **Output**: Per-cell noise averages, dominant sound classes, anomaly counts

### 7. Anomaly Detection Layer
- **Method**: Statistical baseline (z-score) + optional Isolation Forest
- **Purpose**: Identifies acoustically unusual events based on location, time, and historical baseline

### 8. API Layer
- **Framework**: FastAPI
- **Protocols**: REST + WebSocket
- **Purpose**: Serves data to the frontend dashboard

### 9. Frontend Dashboard
- **Technology**: React + MapLibre GL JS
- **Purpose**: Interactive real-time map showing observations, heatmaps, H3 cells, anomalies

---

*This document will be updated as each phase is implemented.*
