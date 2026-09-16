# Real-Time Spatial-Temporal Acoustic Intelligence Platform

## Urban Noise Monitoring, Sound Classification & Anomaly Detection

An academic/research-oriented platform that simulates a network of urban acoustic sensors, processes audio using deep learning, streams observations through a message broker, stores spatial-temporal data, detects acoustic anomalies, and displays results on an interactive real-time map.

---

## Architecture Overview

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

---

## Technology Stack

| Component           | Technology                        |
|---------------------|-----------------------------------|
| Audio Processing    | Librosa, NumPy                    |
| ML Framework        | PyTorch                           |
| Feature Type        | Log-Mel Spectrograms              |
| Streaming           | Apache Kafka                      |
| Database            | PostgreSQL + PostGIS              |
| Spatial Aggregation | H3 (Uber)                         |
| Backend API         | FastAPI                           |
| Frontend            | React + MapLibre GL JS            |
| Real-Time Updates   | WebSocket                         |

---

## Project Structure

```
urban-noise-intelligence/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── data/
│   ├── raw/                    # Place UrbanSound8K here
│   ├── processed/              # Extracted features
│   └── metadata/               # Dataset metadata
├── src/
│   ├── config/                 # Central configuration
│   ├── audio/                  # Audio loading & preprocessing
│   ├── features/               # Feature extraction (mel, log-mel)
│   ├── models/                 # Model architectures
│   ├── training/               # Training loops
│   ├── inference/              # Inference utilities
│   ├── simulation/             # Sensor simulation
│   ├── streaming/              # Kafka producer/consumer
│   ├── database/               # PostgreSQL/PostGIS
│   ├── spatial/                # H3 aggregation
│   ├── anomaly/                # Anomaly detection
│   └── utils/                  # Visualization, helpers
├── backend/                    # FastAPI application
├── frontend/                   # React dashboard
├── models/                     # Saved model checkpoints
├── scripts/                    # Standalone scripts
├── tests/                      # Unit tests
├── results/                    # Evaluation outputs, plots
└── docs/                       # Documentation
```

---

## Quick Start

### 1. Prerequisites

- Python 3.9+ (tested on Windows)
- Git

### 2. Clone and Setup

```bash
git clone <repository-url>
cd urban-noise-intelligence

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Dataset Setup

Download **UrbanSound8K** from: https://urbansounddataset.weebly.com/urbansound8k.html

Extract it so the structure looks like:

```
data/raw/UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   ...
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

See `data/raw/README.md` for detailed instructions.

### 4. Run Audio Exploration (Phase 1)

```bash
python scripts/explore_audio.py
```

This will load a sample audio file, extract features, and display visualizations.

---

## Development Phases

| Phase | Description                 | Status      |
|-------|-----------------------------|-------------|
| 0     | Project Setup               | ✅ Complete |
| 1     | Audio Processing            | 🔄 Current  |
| 2     | CNN Sound Classification    | ⬜ Pending  |
| 3     | Model Evaluation            | ⬜ Pending  |
| 4     | Sensor Simulation           | ⬜ Pending  |
| 5     | Kafka Streaming             | ⬜ Pending  |
| 6     | PostgreSQL + PostGIS        | ⬜ Pending  |
| 7     | FastAPI Backend             | ⬜ Pending  |
| 8     | H3 Spatial Aggregation      | ⬜ Pending  |
| 9     | Anomaly Detection           | ⬜ Pending  |
| 10    | Interactive Map Dashboard   | ⬜ Pending  |
| 11    | Real-Time WebSocket         | ⬜ Pending  |
| 12    | Analytics + Integration     | ⬜ Pending  |
| 13    | Documentation + Experiments | ⬜ Pending  |

---

## Research Questions

1. Can deep-learning audio classification identify common urban acoustic events from crowdsourced audio?
2. How does sensor/device calibration affect acoustic measurements and downstream analysis?
3. Can an event-driven architecture process simulated acoustic observations with low enough latency for near-real-time visualization?
4. How does spatial aggregation resolution affect the representation of urban acoustic patterns?
5. Can historical spatial-temporal baselines identify unusual acoustic events more effectively than a simple fixed noise threshold?
6. What are the main limitations when moving from controlled datasets to real-world crowdsourced urban audio?

---

## License

This project is for academic/research purposes. The UrbanSound8K dataset has its own licensing terms — see its official page for details.
