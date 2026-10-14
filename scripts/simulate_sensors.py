"""
Sensor Simulation CLI Script — Phase 4

Simulates an urban fleet of smartphone and IoT acoustic sensors moving
through distinct urban zones (residential, highway, market, construction, park).

Emits structured observation events conforming to Master Prompt Section 16 schema.

Usage:
    python scripts/simulate_sensors.py --num-sensors 5 --events 15 --interval 0.5
    python scripts/simulate_sensors.py --events 20 --output results/simulated_events.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.simulation.manager import SimulationFleet

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def format_event_log(event: dict) -> str:
    """Format an observation event into a clear terminal log entry."""
    dev = event["device_id"]
    model = event["device_model"]
    s_class = event["sound_class"]
    conf = event["confidence"] * 100.0
    raw_db = event["noise_level_db"]
    cal_db = event["calibrated_db"]
    offset = event["calibration_offset"]
    sign = "+" if offset >= 0 else ""
    lat = event["latitude"]
    lon = event["longitude"]
    acc = event["gps_accuracy"]

    return (
        f"[SENSOR] {dev:<10} ({model:<22}) @ [{lat:.5f}, {lon:.5f}] (+/- {acc}m)\n"
        f"  --> ML: {s_class.upper():<16} (conf: {conf:4.1f}%)\n"
        f"  --> SPL: {raw_db:4.1f} dB (calibrated: {cal_db:4.1f} dB, offset: {sign}{offset:0.1f} dB)"
    )


def main():
    parser = argparse.ArgumentParser(description="Simulate urban acoustic sensor network.")
    parser.add_argument("--num-sensors", type=int, default=5, help="Number of simulated sensing nodes")
    parser.add_argument("--events", type=int, default=10, help="Total observation events to generate")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds delay between observations")
    parser.add_argument("--output", type=str, default="results/simulated_events.json", help="Path to save emitted events")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 4: SIMULATED ACOUSTIC SENSING FLEET")
    print("=" * 70)
    print(f"  Active Sensors:     {args.num_sensors}")
    print(f"  Total Observations: {args.events}")
    print(f"  Interval:           {args.interval:.1f} s")
    print(f"  Output Artifact:    {args.output}")
    print("-" * 70)

    fleet = SimulationFleet(num_sensors=args.num_sensors)

    print("REGISTERED SENSOR FLEET:")
    for s in fleet.sensors:
        sign = "+" if s.calibration_offset >= 0 else ""
        print(
            f"  * {s.device_id}: {s.device_model:<25} "
            f"Offset: {sign}{s.calibration_offset:0.1f} dB | "
            f"Zone: {s.current_zone.name if s.current_zone else 'Unassigned'}"
        )
    print("=" * 70)
    print("[INFO] Starting real-time sensor observation stream...")
    print()

    collected_events = []

    def on_event(ev: dict):
        collected_events.append(ev)
        print(format_event_log(ev))
        print("-" * 70)

    fleet.run_simulation(
        num_events=args.events,
        interval_sec=args.interval,
        callback=on_event,
    )

    # Save events to output JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(collected_events, f, indent=2)

    print()
    print("=" * 70)
    print(f"[SUCCESS] Stream complete! {len(collected_events)} events written to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
