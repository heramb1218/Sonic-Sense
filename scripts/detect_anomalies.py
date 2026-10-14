"""
Acoustic Anomaly Detection CLI Script (Phase 9)

Detects and analyzes acoustic anomalies in the urban noise sensor network:
- Statistical z-score deviations against baseline sound levels
- Domain-specific context violations (nighttime construction, sirens/gunshots in quiet zones)
- Unsupervised machine learning (Isolation Forest)
- Generates ASCII terminal reports and exports flagged anomalies to JSON

Usage:
    # Run anomaly evaluation on recent database records:
    python scripts/detect_anomalies.py

    # Filter with higher anomaly score threshold (e.g. 2.5) and export to JSON:
    python scripts/detect_anomalies.py --min-score 2.5 --output results/anomalies.json

    # Train Isolation Forest on historical data and simulate live test events:
    python scripts/detect_anomalies.py --simulate --train-ml
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import get_db_manager
from src.database.repository import ObservationRepository
from src.anomaly.detector import AcousticAnomalyDetector
from src.simulation.manager import SimulationFleet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect acoustic anomalies across urban noise sensor observations."
    )
    parser.add_argument(
        "--min-score", "-s",
        type=float,
        default=2.0,
        help="Minimum anomaly score threshold to display (default: 2.0)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=200,
        help="Maximum recent observations to evaluate (default: 200)",
    )
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Train an Isolation Forest unsupervised model on historical data",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional path to export detected anomalies to JSON (e.g. results/anomalies.json)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate sensor fleet observations first to ensure diverse events",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 85)
    print("ACOUSTIC ANOMALY DETECTION ENGINE (PHASE 9)")
    print("=" * 85)
    print(f"Minimum Anomaly Score: {args.min_score}")
    print(f"Observation Evaluation Limit: {args.limit}")

    db = get_db_manager()
    db.init_db()
    repo = ObservationRepository(db_manager=db)

    # Simulate fleet observations if requested or if database is empty
    recent_records = repo.get_latest_observations(limit=args.limit)
    if not recent_records or args.simulate:
        print("\n[INFO] Simulating sensor fleet observations to ensure diverse acoustic scenarios...")
        fleet = SimulationFleet(num_sensors=10)
        events = fleet.run_simulation(num_events=20, interval_sec=0)
        import uuid
        events.append({
            "event_id": f"SIM-ANOMALY-EXPLOSION-{uuid.uuid4().hex[:8]}",
            "device_id": "TEST-SMARTPHONE-999",
            "timestamp": "2026-10-14T23:30:00Z",
            "latitude": 18.5024,
            "longitude": 73.8388,  # Residential/Park Zone
            "noise_level_db": 94.5,
            "sound_class": "siren",
            "confidence": 0.96,
            "calibration_offset": 0.0,
            "calibrated_db": 94.5,
        })
        repo.insert_batch(events)
        print(f"[INFO] Ingested {len(events)} events (including high-decibel disturbance).")
        recent_records = repo.get_latest_observations(limit=args.limit)

    print(f"[INFO] Total observations collected for evaluation: {len(recent_records)}")

    # Initialize anomaly detector
    detector = AcousticAnomalyDetector(z_moderate_threshold=args.min_score)

    if args.train_ml and len(recent_records) >= 10:
        print("[INFO] Training Isolation Forest on historical acoustic vectors...")
        detector.train_isolation_forest(recent_records)
        print("[INFO] Isolation Forest trained successfully.")

    # Evaluate each observation
    flagged_anomalies = []
    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "NORMAL": 0}

    for obs in recent_records:
        eval_result = detector.evaluate_observation(obs)
        severity = eval_result["severity"]
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

        if eval_result["anomaly_score"] >= args.min_score or eval_result["is_anomaly"]:
            flagged_anomalies.append({
                "event_id": obs.event_id,
                "timestamp": str(obs.timestamp),
                "latitude": obs.latitude,
                "longitude": obs.longitude,
                "sound_class": obs.sound_class,
                "confidence": obs.confidence,
                "calibrated_db": obs.calibrated_db,
                "anomaly_score": eval_result["anomaly_score"],
                "severity": eval_result["severity"],
                "reasons": eval_result["reasons"],
                "z_score": eval_result["z_score"],
                "baseline_mean": eval_result["baseline_mean"],
                "zone_type": eval_result["zone_type"],
            })

    print(f"\n[SUMMARY] Evaluated: {len(recent_records)} | Anomalies Detected: {len(flagged_anomalies)}")
    print(
        f"          Critical: {severity_counts.get('CRITICAL', 0)} | "
        f"High: {severity_counts.get('HIGH', 0)} | "
        f"Moderate: {severity_counts.get('MODERATE', 0)} | "
        f"Normal: {severity_counts.get('NORMAL', 0)}\n"
    )

    # Format table output (safe ASCII for Windows console)
    header = (
        f"{'Severity':<9} | {'Score':<5} | {'dB':<5} | {'Class':<14} | "
        f"{'Zone':<12} | {'Reasons'}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Sort flagged anomalies by score descending
    sorted_anomalies = sorted(flagged_anomalies, key=lambda a: a["anomaly_score"], reverse=True)

    for a in sorted_anomalies[:25]:
        reasons_summary = "; ".join(a["reasons"][:2]) if a["reasons"] else "Statistical deviation"
        print(
            f"{a['severity']:<9} | {a['anomaly_score']:<5.1f} | {a['calibrated_db']:<5.1f} | "
            f"{a['sound_class']:<14} | {a['zone_type']:<12} | {reasons_summary}"
        )

    if len(sorted_anomalies) > 25:
        print(f"... and {len(sorted_anomalies) - 25} more anomalies.")
    print("-" * len(header))

    # Export to JSON if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(flagged_anomalies, f, indent=2)
        print(f"\n[SUCCESS] Exported {len(flagged_anomalies)} anomalies to:")
        print(f"          {out_path.resolve()}")

    print("\n[SUCCESS] Acoustic Anomaly Detection Completed.")


if __name__ == "__main__":
    main()
