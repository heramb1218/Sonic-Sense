"""
End-to-End System Verification & Pipeline Diagnostic Audit (Phase 12)

Audits and demonstrates all 9 architectural stages of the platform:
Stage 1: Audio Ingestion & Log-Mel Spectrogram Extraction
Stage 2: PyTorch CNN Sound Classification Inference
Stage 3: Sensor Hardware Calibration & GPS Mobility
Stage 4: Message Broker Publication (Kafka / Local Queue)
Stage 5: Stream Consumer Ingestion & FIFO Processing
Stage 6: Multi-Tier Acoustic Anomaly Detection (Z-Score + Rules)
Stage 7: Spatial-Temporal Database Persistence & Spatial Queries
Stage 8: Uber H3 Hexagonal Spatial Aggregation & GeoJSON Polygons
Stage 9: REST API & Real-Time WebSocket Live Delivery

Usage:
    python scripts/demo_end_to_end.py
"""

import sys
import time
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.audio.processor import load_audio, preprocess_audio
from src.features.extractor import extract_log_mel_spectrogram
from src.inference.predict import SoundClassifier
from src.simulation.zones import ZoneRegistry
from src.simulation.sensors import SimulatedSensor
from src.streaming.broker import get_broker
from src.streaming.producer import NoiseEventProducer
from src.streaming.consumer import NoiseEventConsumer
from src.anomaly.detector import AcousticAnomalyDetector
from src.database.connection import get_db_manager
from src.database.repository import ObservationRepository
from src.spatial.h3_grid import H3GridManager
from backend.main import app
from fastapi.testclient import TestClient


def main():
    print("=" * 85)
    print("URBAN ACOUSTIC INTELLIGENCE -- END-TO-END SYSTEM INTEGRATION AUDIT")
    print("=" * 85)
    print(f"Timestamp : {datetime.now(timezone.utc).isoformat()}")
    print(f"Python    : {sys.version.split()[0]}")
    print("=" * 85)

    audit_results = []
    total_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Stage 1: Audio Ingestion & Feature Extraction
    # -------------------------------------------------------------------------
    sample_file = settings.DATA_DIR / "sample" / "sample_siren_urban.wav"
    if not sample_file.exists():
        # Generate on the fly if needed
        import soundfile as sf
        from scripts.generate_sample_audio import generate_synthetic_siren
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        audio_syn = generate_synthetic_siren(duration=settings.AUDIO_DURATION, sr=settings.SAMPLE_RATE)
        sf.write(str(sample_file), audio_syn, settings.SAMPLE_RATE)

    audio, sr = load_audio(sample_file)
    # Warm-up feature extraction (avoids cold-start JIT compilation latency)
    _ = extract_log_mel_spectrogram(preprocess_audio(audio[:sr], sr=sr), sr=sr)

    t0 = time.perf_counter()
    processed_audio = preprocess_audio(audio, sr=sr)
    log_mel = extract_log_mel_spectrogram(processed_audio, sr=sr)
    latency_ms = (time.perf_counter() - t0) * 1000

    stage1_pass = (log_mel.ndim == 2 and log_mel.shape[0] == 128 and latency_ms < 1500)
    audit_results.append({
        "stage": 1,
        "name": "Audio Preprocessing & Log-Mel Extraction",
        "passed": stage1_pass,
        "latency_ms": round(latency_ms, 2),
        "details": f"Sample rate: {sr} Hz, Shape: {log_mel.shape}, Range: [{log_mel.min():.1f}, {log_mel.max():.1f}] dB",
    })
    print(f"[{'PASS' if stage1_pass else 'FAIL'}] Stage 1: Audio Processing & Log-Mel Extraction ({latency_ms:.1f}ms) -- Shape: {log_mel.shape}")

    # -------------------------------------------------------------------------
    # Stage 2: Deep Learning Sound Classifier
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        classifier = SoundClassifier()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[WARNING] Production model not available ({exc}). Running audit with explicitly permitted development baseline weights.")
        classifier = SoundClassifier(allow_untrained=True)
    pred = classifier.predict_audio(processed_audio, sr=sr)
    latency_ms = (time.perf_counter() - t0) * 1000

    stage2_pass = (pred["predicted_class"] is not None and 0.0 <= pred["confidence"] <= 1.0)
    audit_results.append({
        "stage": 2,
        "name": "Deep Learning Sound Classification",
        "passed": stage2_pass,
        "latency_ms": round(latency_ms, 2),
        "details": f"Predicted: '{pred['predicted_class']}' ({pred['confidence']*100:.1f}% confidence)",
    })
    print(f"[{'PASS' if stage2_pass else 'FAIL'}] Stage 2: PyTorch CNN Inference ({latency_ms:.1f}ms) -- Predicted: '{pred['predicted_class']}' ({pred['confidence']*100:.1f}%)")

    # -------------------------------------------------------------------------
    # Stage 3: Simulated Sensor Calibration & Mobility
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    registry = ZoneRegistry()
    zone = registry.get_all_zones()[0]
    sensor = SimulatedSensor(
        device_id="AUDIT-SENSOR-001",
        device_model="Apple iPhone 14 Pro",
        calibration_offset=-1.8,
        current_zone=zone,
    )
    sensor.step(dt_seconds=2.0)
    event_payload = sensor.capture_observation(
        audio_file_path=sample_file,
        classifier=classifier,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    stage3_pass = (
        event_payload["device_id"] == "AUDIT-SENSOR-001"
        and event_payload["calibrated_db"] is not None
        and event_payload["h3_cell"] is not None
    )
    audit_results.append({
        "stage": 3,
        "name": "Sensor Hardware Calibration & Mobility",
        "passed": stage3_pass,
        "latency_ms": round(latency_ms, 2),
        "details": f"Node at ({event_payload['latitude']:.4f}, {event_payload['longitude']:.4f}), Calibrated dB: {event_payload['calibrated_db']:.1f}",
    })
    print(f"[{'PASS' if stage3_pass else 'FAIL'}] Stage 3: Sensor Calibration & Mobility ({latency_ms:.1f}ms) -- Calibrated: {event_payload['calibrated_db']:.1f} dB (H3: {event_payload['h3_cell']})")

    # -------------------------------------------------------------------------
    # Stage 4: Message Broker Publication
    # -------------------------------------------------------------------------
    broker = get_broker()
    topic = f"audit-stream-{uuid.uuid4().hex[:6]}"
    producer = NoiseEventProducer(broker=broker, topic=topic)
    t0 = time.perf_counter()
    msg = producer.send_event(event_payload, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000

    stage4_pass = (msg is not None and msg.topic == topic)
    audit_results.append({
        "stage": 4,
        "name": "Message Broker Stream Publication",
        "passed": stage4_pass,
        "latency_ms": round(latency_ms, 2),
        "details": f"Topic: '{topic}', Broker Offset: {msg.offset}",
    })
    print(f"[{'PASS' if stage4_pass else 'FAIL'}] Stage 4: Message Broker Publication ({latency_ms:.1f}ms) -- Topic: '{topic}' (Offset: {msg.offset})")

    # -------------------------------------------------------------------------
    # Stage 5: Stream Consumer Processing
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    consumer = NoiseEventConsumer(broker=broker, topic=topic)
    consumed_events = list(consumer.consume_events(max_messages=1, timeout=0.5, verbose=False))
    latency_ms = (time.perf_counter() - t0) * 1000

    stage5_pass = (len(consumed_events) == 1 and consumed_events[0]["event_id"] == event_payload["event_id"])
    audit_results.append({
        "stage": 5,
        "name": "Stream Consumer Ingestion & FIFO Ordering",
        "passed": stage5_pass,
        "latency_ms": round(latency_ms, 2),
        "details": f"Consumed event ID: {consumed_events[0]['event_id'] if consumed_events else 'None'}",
    })
    print(f"[{'PASS' if stage5_pass else 'FAIL'}] Stage 5: Stream Consumer Processing ({latency_ms:.1f}ms) -- Received: 1 event")

    # -------------------------------------------------------------------------
    # Stage 6: Acoustic Anomaly Detection
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    detector = AcousticAnomalyDetector()
    eval_res = detector.evaluate_observation(event_payload)
    latency_ms = (time.perf_counter() - t0) * 1000

    stage6_pass = ("severity" in eval_res and "anomaly_score" in eval_res and "reasons" in eval_res)
    audit_results.append({
        "stage": 6,
        "name": "Multi-Factor Acoustic Anomaly Detection",
        "passed": stage6_pass,
        "latency_ms": round(latency_ms, 2),
        "details": f"Severity: {eval_res['severity']}, Score: {eval_res['anomaly_score']}, Reasons: {len(eval_res['reasons'])}",
    })
    print(f"[{'PASS' if stage6_pass else 'FAIL'}] Stage 6: Anomaly Detection Engine ({latency_ms:.1f}ms) -- Severity: {eval_res['severity']} (Score: {eval_res['anomaly_score']})")

    # -------------------------------------------------------------------------
    # Stage 7: Spatial-Temporal Database Persistence
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    db = get_db_manager()
    db.init_db()
    repo = ObservationRepository(db_manager=db)
    obs = repo.insert_observation(event_payload)
    latest = repo.get_latest_observations(limit=5)
    latency_ms = (time.perf_counter() - t0) * 1000

    stage7_pass = (obs.id is not None and len(latest) >= 1)
    audit_results.append({
        "stage": 7,
        "name": "Spatial-Temporal Database Persistence",
        "passed": stage7_pass,
        "latency_ms": round(latency_ms, 2),
        "details": f"Dialect: {db.engine.dialect.name.upper()}, Saved Record ID: {obs.id}",
    })
    print(f"[{'PASS' if stage7_pass else 'FAIL'}] Stage 7: Database Persistence & Queries ({latency_ms:.1f}ms) -- Record ID: {obs.id}")

    # -------------------------------------------------------------------------
    # Stage 8: Uber H3 Hexagonal Spatial Aggregation
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    hexagons = H3GridManager.aggregate_acoustic_observations(latest, resolution=8)
    geojson_fc = H3GridManager.to_geojson_feature_collection(hexagons)
    latency_ms = (time.perf_counter() - t0) * 1000

    stage8_pass = (len(hexagons) >= 1 and geojson_fc["type"] == "FeatureCollection")
    audit_results.append({
        "stage": 8,
        "name": "Uber H3 Hexagonal Spatial Aggregation",
        "passed": stage8_pass,
        "latency_ms": round(latency_ms, 2),
        "details": f"Cells: {len(hexagons)}, GeoJSON features: {len(geojson_fc['features'])}",
    })
    print(f"[{'PASS' if stage8_pass else 'FAIL'}] Stage 8: Uber H3 Hexagonal Aggregation ({latency_ms:.1f}ms) -- Mapped into {len(hexagons)} cell(s)")

    # -------------------------------------------------------------------------
    # Stage 9: REST API & Live WebSocket Delivery
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    client = TestClient(app)
    api_payload = dict(event_payload)
    api_payload["event_id"] = str(uuid.uuid4())
    with client.websocket_connect("/ws/live") as ws:
        api_post_res = client.post("/observations", json=api_payload)
        assert api_post_res.status_code == 201
        ws_msg = ws.receive_json()
    latency_ms = (time.perf_counter() - t0) * 1000

    stage9_pass = (ws_msg["event_id"] == api_payload["event_id"])
    audit_results.append({
        "stage": 9,
        "name": "REST API & Real-Time WebSocket Delivery",
        "passed": stage9_pass,
        "latency_ms": round(latency_ms, 2),
        "details": f"POST HTTP 201, WebSocket received broadcast matching event_id",
    })
    print(f"[{'PASS' if stage9_pass else 'FAIL'}] Stage 9: REST API & WebSocket Broadcast ({latency_ms:.1f}ms) -- Broadcast delivered")

    total_latency_ms = (time.perf_counter() - total_start) * 1000
    passed_stages = sum(1 for s in audit_results if s["passed"])

    print("-" * 85)
    print(f"OVERALL PIPELINE INTEGRITY : {passed_stages}/{len(audit_results)} STAGES PASSED ({passed_stages/len(audit_results)*100:.1f}%)")
    print(f"TOTAL END-TO-END LATENCY  : {total_latency_ms:.1f} ms")
    print("-" * 85)

    # Export audit results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    audit_file = results_dir / "pipeline_audit.json"
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "passed_stages": passed_stages,
            "total_stages": len(audit_results),
            "total_latency_ms": round(total_latency_ms, 2),
            "stages": audit_results,
        }, f, indent=2)

    print(f"[SUCCESS] Audit report exported to: {audit_file.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    main()
