"""
Master Platform Launcher — Urban Acoustic Intelligence Platform (Phase 12)

Launches the complete unified platform in a single command:
1. Initializes spatial-temporal database schema (PostgreSQL+PostGIS / SQLite)
2. Starts background sensor simulation streaming thread (producing realistic observations into the broker)
3. Boots FastAPI web server hosting REST APIs, WebSocket live hub, and the dark-mode dashboard
4. Serves the interactive GIS map and dashboard directly at http://127.0.0.1:8000/

Usage:
    # Launch entire platform with live sensor simulation:
    python scripts/run_platform.py

    # Launch on custom port or with custom sensor count:
    python scripts/run_platform.py --port 8000 --sensors 10 --interval 1.5

    # Launch server only (without background sensor simulation):
    python scripts/run_platform.py --no-simulation
"""

import sys
import time
import json
import argparse
import threading
import logging
import urllib.request
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from src.config.settings import settings
from src.database.connection import get_db_manager
from src.simulation.manager import SimulationFleet
from src.anomaly.detector import AcousticAnomalyDetector

logger = logging.getLogger("platform_launcher")


class SimulationStreamerThread(threading.Thread):
    """Background worker continuously generating sensor fleet events and broadcasting to the dashboard."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, num_sensors: int = 8, interval_sec: float = 2.0):
        super().__init__(daemon=True, name="AcousticSensorStreamer")
        self.api_url = f"http://{host}:{port}/observations"
        self.num_sensors = num_sensors
        self.interval_sec = interval_sec
        self.stop_signal = threading.Event()
        self.fleet = SimulationFleet(num_sensors=self.num_sensors)
        self.detector = AcousticAnomalyDetector()

    def run(self):
        logger.info(f"Background sensor streamer started ({self.num_sensors} mobile nodes, {self.interval_sec}s interval).")
        # Give Uvicorn 2 seconds to initialize before emitting events
        time.sleep(2.0)

        while not self.stop_signal.is_set():
            try:
                # 1. Generate observation event from moving sensor
                raw_event = self.fleet.generate_next_event()

                # 2. Evaluate acoustic anomaly
                eval_res = self.detector.evaluate_observation(raw_event)
                raw_event["anomaly_score"] = eval_res["anomaly_score"]
                raw_event["anomaly_severity"] = eval_res["severity"]
                raw_event["anomaly_reasons"] = eval_res["reasons"]

                # 3. Post to REST API which persists and broadcasts natively via WebSocket
                data = json.dumps(raw_event, default=str).encode("utf-8")
                req = urllib.request.Request(
                    self.api_url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    pass

            except Exception as e:
                logger.debug(f"Simulation streamer waiting for API ({e})")

            self.stop_signal.wait(self.interval_sec)

        logger.info("Background sensor streamer stopped cleanly.")

    def stop(self):
        self.stop_signal.set()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch complete Urban Acoustic Intelligence Platform."
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.BACKEND_HOST,
        help=f"Server host address (default: {settings.BACKEND_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.BACKEND_PORT,
        help=f"Server port (default: {settings.BACKEND_PORT})",
    )
    parser.add_argument(
        "--sensors", "-s",
        type=int,
        default=8,
        help="Number of simulated mobile sensors (default: 8)",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=2.0,
        help="Seconds between sensor emissions (default: 2.0s)",
    )
    parser.add_argument(
        "--no-simulation",
        action="store_true",
        help="Disable background sensor fleet simulation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db = get_db_manager()
    db.init_db()

    dialect_name = db.engine.dialect.name.upper()

    print("=" * 85)
    print("URBAN ACOUSTIC INTELLIGENCE PLATFORM -- MASTER LAUNCHER (PHASE 12)")
    print("=" * 85)
    print(f"Interactive Dashboard : http://{args.host}:{args.port}/")
    print(f"API Documentation     : http://{args.host}:{args.port}/docs")
    print(f"Database Engine       : {dialect_name} ({db.db_url})")
    print(f"Streaming Mode        : {settings.STREAMING_MODE.upper()}")
    print(f"Sensor Fleet Status   : {'Disabled (--no-simulation)' if args.no_simulation else f'Active ({args.sensors} nodes, {args.interval}s interval)'}")
    print("=" * 85)
    print("Press Ctrl+C to gracefully stop the platform.\n")

    streamer_thread = None
    if not args.no_simulation:
        streamer_thread = SimulationStreamerThread(
            host=args.host,
            port=args.port,
            num_sensors=args.sensors,
            interval_sec=args.interval,
        )
        streamer_thread.start()

    try:
        uvicorn.run(
            "backend.main:app",
            host=args.host,
            port=args.port,
            log_level="info",
            reload=False,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Platform shutdown initiated...")
    finally:
        if streamer_thread is not None:
            print("[INFO] Stopping background sensor fleet streamer...")
            streamer_thread.stop()
            streamer_thread.join(timeout=3.0)
        print("[SUCCESS] Urban Acoustic Intelligence Platform stopped.")


if __name__ == "__main__":
    main()
