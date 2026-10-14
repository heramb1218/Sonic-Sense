"""
Live Streaming Demonstration Script — Real-Time Acoustic Stream (Phase 11)

Demonstrates the complete end-to-end live streaming pipeline (Section 24):
1. Simulated sensor fleet moving through urban zones generating acoustic events (interval: 1-2 sec)
2. Events streamed through the message broker (Kafka or local fallback)
3. Consumer process reading events, detecting anomalies, and persisting to database
4. WebSocket broadcasting new observations to connected frontend clients
5. Live dashboard map updates sensor locations, decibel meter, and alert notifications

Usage:
    # Run 20 streaming events at 1.0 second intervals:
    python scripts/run_live_stream.py --events 20 --interval 1.0

    # Continuous streaming to backend server on port 8000:
    python scripts/run_live_stream.py --interval 1.5 --post-api

    # High-speed simulation test (10 events at 0.1 sec interval):
    python scripts/run_live_stream.py --events 10 --interval 0.1
"""

import sys
import time
import argparse
import asyncio
from pathlib import Path
import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.simulation.manager import SimulationFleet
from src.streaming.broker import get_broker
from src.streaming.producer import NoiseEventProducer
from src.streaming.consumer import NoiseEventConsumer
from src.anomaly.detector import AcousticAnomalyDetector
from src.database.connection import get_db_manager
from src.database.repository import ObservationRepository
from backend.websocket.connection_manager import ws_manager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run real-time acoustic intelligence live streaming pipeline."
    )
    parser.add_argument(
        "--sensors", "-s",
        type=int,
        default=8,
        help="Number of mobile simulated sensor nodes (default: 8)",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=1.5,
        help="Streaming interval in seconds between events (default: 1.5s)",
    )
    parser.add_argument(
        "--events", "-n",
        type=int,
        default=30,
        help="Total events to stream (0 for continuous until Ctrl+C, default: 30)",
    )
    parser.add_argument(
        "--post-api",
        action="store_true",
        help="Post events via HTTP to running FastAPI backend on http://127.0.0.1:8000/observations",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 85)
    print("URBAN ACOUSTIC INTELLIGENCE -- REAL-TIME LIVE STREAMING PIPELINE")
    print("=" * 85)
    print(f"Active Sensor Fleet : {args.sensors} nodes")
    print(f"Emission Interval   : {args.interval}s")
    print(f"Total Stream Events : {'Continuous (Ctrl+C to stop)' if args.events == 0 else args.events}")
    print(f"API Forwarding Mode : {'HTTP POST /observations' if args.post_api else 'Local In-Process WebSocket Hub'}")
    print("=" * 85)

    # 1. Initialize Pipeline Infrastructure
    db = get_db_manager()
    db.init_db()
    repo = ObservationRepository(db_manager=db)
    detector = AcousticAnomalyDetector()
    broker = get_broker()
    producer = NoiseEventProducer(broker=broker)
    consumer = NoiseEventConsumer(broker=broker, anomaly_detector=detector)
    fleet = SimulationFleet(num_sensors=args.sensors)

    backend_api_url = f"http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}/observations"

    count = 0
    anomalies_detected = 0

    try:
        while True:
            count += 1
            # Step 1: Simulated Sensor Captures Acoustic Observation
            raw_event = fleet.generate_next_event()
            node_id = raw_event["device_id"]
            db_val = raw_event["calibrated_db"]
            sound_class = raw_event["sound_class"]
            lat, lon = raw_event["latitude"], raw_event["longitude"]

            print(f"\n[STREAM #{count:03d}] Node: {node_id} at ({lat:.4f}, {lon:.4f}) | Sound: {sound_class} ({db_val:.1f} dB)")

            # Step 2: Stream Event via Broker (Kafka / LocalQueue)
            producer.send_event(raw_event, verbose=False)
            print(f"  --> [BROKER] Published to topic '{settings.KAFKA_TOPIC_EVENTS}'")

            # Step 3: Consumer Reads, Evaluates Anomaly, and Persists to Database
            consumed_messages = list(consumer.consume_events(max_messages=1, timeout=0.5, verbose=False))
            if consumed_messages:
                processed_event = consumed_messages[0]
                eval_res = detector.evaluate_observation(processed_event)

                processed_event["anomaly_score"] = eval_res["anomaly_score"]
                processed_event["anomaly_severity"] = eval_res["severity"]
                processed_event["anomaly_reasons"] = eval_res["reasons"]

                # Persist to database
                repo.insert_observation(processed_event)

                if eval_res["is_anomaly"]:
                    anomalies_detected += 1
                    reasons_summary = "; ".join(eval_res["reasons"])
                    print(f"  --> [ANOMALY ALERT] Severity: {eval_res['severity']} (Score: {eval_res['anomaly_score']}) | {reasons_summary}")
                else:
                    print(f"  --> [DATABASE] Stored observation (id: {processed_event['event_id'][:8]}...)")

                # Step 4: Broadcast via WebSocket
                if args.post_api:
                    try:
                        with httpx.Client(timeout=1.0) as client:
                            resp = client.post(backend_api_url, json=processed_event)
                            if resp.status_code == 201:
                                print(f"  --> [WEBSOCKET] Broadcasted via HTTP POST to FastAPI backend ({backend_api_url})")
                            else:
                                print(f"  --> [WARNING] Backend HTTP status: {resp.status_code}")
                    except Exception as e:
                        print(f"  --> [NOTICE] Backend server offline ({e}). Broadcasted locally.")
                        asyncio.run(ws_manager.broadcast_event(processed_event))
                else:
                    # In-process WebSocket broadcast
                    asyncio.run(ws_manager.broadcast_event(processed_event))
                    print(f"  --> [WEBSOCKET] In-process broadcast to {ws_manager.connection_count} active client(s)")

            # Check termination
            if args.events > 0 and count >= args.events:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[INFO] Streaming stopped by user.")

    finally:
        producer.close()
        consumer.close()

    print("\n" + "=" * 85)
    print(f"[SUCCESS] Live Streaming Completed: {count} events streamed | {anomalies_detected} anomalies detected.")
    print("=" * 85)


if __name__ == "__main__":
    main()
