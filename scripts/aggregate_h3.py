"""
Batch Spatial Aggregation Script — Uber H3 Hexagonal Urban Noise Mapping (Phase 8)

Computes spatial aggregations of acoustic measurements across H3 hexagonal cells.
Optionally exports a GeoJSON FeatureCollection ready for web mapping (MapLibre/Leaflet).

Usage:
    # Run aggregation at default resolution (8):
    python scripts/aggregate_h3.py

    # Run at higher resolution (9) with GeoJSON export:
    python scripts/aggregate_h3.py --resolution 9 --output results/h3_hexagons.geojson

    # Simulate sensor data first if database is empty:
    python scripts/aggregate_h3.py --simulate --limit 100
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.database.connection import get_db_manager
from src.database.repository import ObservationRepository
from src.spatial.h3_grid import H3GridManager
from src.simulation.manager import SimulationFleet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate urban acoustic measurements into Uber H3 hexagons."
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=settings.H3_DEFAULT_RESOLUTION,
        help=f"H3 hexagon resolution (default: {settings.H3_DEFAULT_RESOLUTION})",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=500,
        help="Maximum recent observations to process (default: 500)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional path to export GeoJSON FeatureCollection (e.g. results/h3_hexagons.geojson)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate sensor observations first if database is empty or requested",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("UBER H3 SPATIAL HEXAGONAL AGGREGATION ENGINE (PHASE 8)")
    print("=" * 80)
    print(f"Target Resolution: {args.resolution}")
    print(f"Observation Limit: {args.limit}")

    db = get_db_manager()
    db.init_db()
    repo = ObservationRepository(db_manager=db)

    # Check database observation count
    recent_records = repo.get_latest_observations(limit=args.limit)

    if not recent_records or args.simulate:
        print("\n[INFO] Simulating sensor fleet observations to ensure representative spatial coverage...")
        fleet = SimulationFleet(num_sensors=15)
        simulated_events = fleet.run_simulation(num_events=30, interval_sec=0)
        repo.insert_batch(simulated_events)
        print(f"[INFO] Ingested {len(simulated_events)} simulated events into database.")
        recent_records = repo.get_latest_observations(limit=args.limit)

    print(f"[INFO] Total observations collected for aggregation: {len(recent_records)}")

    # Perform H3 aggregation
    cells = H3GridManager.aggregate_acoustic_observations(
        recent_records, resolution=args.resolution
    )

    print(f"[INFO] Mapped observations into {len(cells)} distinct H3 hexagonal cells.\n")

    # Format table output (safe ASCII for Windows console)
    header = (
        f"{'H3 Cell Index':<18} | {'Center (Lat, Lon)':<22} | {'Count':<5} | "
        f"{'Avg dB':<6} | {'Peak dB':<7} | {'Dominant Class':<16} | {'Anomalies':<9}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Sort cells by observation count descending
    sorted_cells = sorted(cells, key=lambda c: c["observation_count"], reverse=True)

    for c in sorted_cells[:20]:
        center_str = f"({c['center_lat']:.4f}, {c['center_lon']:.4f})"
        print(
            f"{c['h3_cell']:<18} | {center_str:<22} | {c['observation_count']:<5} | "
            f"{c['average_db']:<6.1f} | {c['peak_db']:<7.1f} | {c['dominant_sound_class']:<16} | "
            f"{c['anomaly_count']:<9}"
        )

    if len(sorted_cells) > 20:
        print(f"... and {len(sorted_cells) - 20} more cells.")
    print("-" * len(header))

    # Export GeoJSON if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        geojson_fc = H3GridManager.to_geojson_feature_collection(cells)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(geojson_fc, f, indent=2)
        print(f"\n[SUCCESS] Exported {len(geojson_fc['features'])} GeoJSON polygon features to:")
        print(f"          {out_path.resolve()}")

    print("\n[SUCCESS] Uber H3 Spatial Aggregation Completed.")


if __name__ == "__main__":
    main()
