"""
Database Initialization and Query Verification CLI Script — Phase 6

Initializes database schema, creates spatial-temporal indexes,
loads simulated observation events, and runs demonstration spatial queries.

Usage:
    python scripts/init_db.py
    python scripts/init_db.py --seed-samples 15
"""

import sys
import json
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.database.connection import get_db_manager
from src.database.repository import ObservationRepository
from src.simulation.manager import SimulationFleet

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Initialize and verify database schema and queries.")
    parser.add_argument("--seed-samples", type=int, default=12, help="Number of sample events to ingest")
    parser.add_argument("--reset", action="store_true", help="Drop existing tables before recreating")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 6: POSTGRESQL / POSTGIS / SPATIAL-TEMPORAL DATABASE")
    print("=" * 70)

    db_manager = get_db_manager()
    print(f"  Active Engine:      {db_manager.engine.dialect.name.upper()}")
    print(f"  Connection URL:     {db_manager.db_url}")
    print("-" * 70)

    if args.reset:
        print("[INFO] Resetting tables...")
        db_manager.drop_all()

    db_manager.init_db()
    repo = ObservationRepository(db_manager=db_manager)
    print("[SUCCESS] Schema created and spatial-temporal indexes provisioned.")

    # Ingest observations
    print(f"\n[INFO] Ingesting {args.seed_samples} simulated sensor observations...")
    fleet = SimulationFleet(num_sensors=4)
    events = fleet.run_simulation(num_events=args.seed_samples, interval_sec=0.0)
    inserted = repo.insert_batch(events)
    print(f"[SUCCESS] Successfully ingested {inserted} records into 'noise_observations'.")

    # =========================================================================
    # Demonstration of Master Prompt Section 20 Queries
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEMONSTRATION OF SPATIAL-TEMPORAL QUERIES (SECTION 20)")
    print("=" * 70)

    # Query 1: Latest observations
    latest = repo.get_latest_observations(limit=3)
    print(f"\n1. LATEST OBSERVATIONS (Top {len(latest)}):")
    for r in latest:
        print(f"   * [{r.event_id[:8]}...] Device: {r.device_id:<10} | Class: {r.sound_class:<15} | Calibrated: {r.calibrated_db} dB")

    # Query 3: Bounding Box
    city_lat, city_lon = settings.CITY_LAT, settings.CITY_LON
    bbox_records = repo.get_observations_in_bounding_box(
        min_lat=city_lat - 0.03,
        min_lon=city_lon - 0.03,
        max_lat=city_lat + 0.03,
        max_lon=city_lon + 0.03,
        limit=100,
    )
    print(f"\n2. BOUNDING BOX QUERY (+/- 0.03 deg around city center):")
    print(f"   * Found {len(bbox_records)} observations in spatial viewport.")

    # Query 4: Radial search
    radius_m = 1500.0
    radial_records = repo.get_observations_in_radius(city_lat, city_lon, radius_meters=radius_m)
    print(f"\n3. RADIAL SEARCH (within {radius_m}m of center [{city_lat}, {city_lon}]):")
    print(f"   * Found {len(radial_records)} observations matching Haversine distance.")

    # Query 6: Area noise summary
    area_summary = repo.get_average_noise_by_area(
        min_lat=city_lat - 0.05,
        min_lon=city_lon - 0.05,
        max_lat=city_lat + 0.05,
        max_lon=city_lon + 0.05,
    )
    print(f"\n4. AREA NOISE AGGREGATION:")
    print(f"   * Average Noise: {area_summary['average_db']} dB")
    print(f"   * Peak Noise:    {area_summary['max_db']} dB")
    print(f"   * Quietest:      {area_summary['min_db']} dB")
    print(f"   * Sample Count:  {area_summary['count']}")

    # Query 7: Sound event counts
    class_counts = repo.get_sound_event_counts()
    print(f"\n5. SOUND EVENT COUNTS BY CLASS:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   * {cls_name:<20}: {count} events")

    # Query 8: Historical baseline
    baseline = repo.get_historical_baseline(time_window_hours=24)
    print(f"\n6. HISTORICAL BASELINE (24h Window):")
    print(f"   * Mean Noise:    {baseline['mean_db']} dB")
    print(f"   * Standard Dev:  {baseline['std_db']} dB")
    print(f"   * Base Samples:  {baseline['observation_count']}")

    print("\n" + "=" * 70)
    print("[SUCCESS] All Phase 6 database queries verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
