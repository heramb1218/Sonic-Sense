"""
Streaming Pipeline Demonstration Script — Phase 5

Demonstrates end-to-end event streaming architecture matching Section 18:
Multiple simulated sensors -> Kafka topic -> Consumer -> Verification

Logs displayed:
[SENSOR] sensor_003 generated observation
[KAFKA] Event published
[CONSUMER] Event received
[ML] Prediction = siren

Usage:
    python scripts/demo_streaming_pipeline.py --events 5
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.simulation.manager import SimulationFleet
from src.streaming.producer import NoiseEventProducer
from src.streaming.consumer import NoiseEventConsumer
from src.streaming.broker import get_broker

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Demonstrate Kafka Event Streaming Pipeline.")
    parser.add_argument("--events", type=int, default=6, help="Number of events to produce and consume")
    parser.add_argument("--sensors", type=int, default=4, help="Number of simulated sensing nodes")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 5: EVENT-DRIVEN STREAMING DEMONSTRATION")
    print("=" * 70)
    print(f"  Target Topic:    {settings.KAFKA_TOPIC_EVENTS}")
    print(f"  Consumer Group:  {settings.KAFKA_CONSUMER_GROUP}")
    print(f"  Total Events:    {args.events}")
    print(f"  Simulated Fleet: {args.sensors} sensors")
    print("-" * 70)

    # Initialize shared broker, producer, and consumer
    broker = get_broker()
    producer = NoiseEventProducer(broker=broker)
    consumer = NoiseEventConsumer(broker=broker)
    fleet = SimulationFleet(num_sensors=args.sensors)

    print()
    print("[INFO] Starting real-time event streaming pipeline...")
    print("=" * 70)

    # Stream and process events
    for i in range(1, args.events + 1):
        print(f"\n--- [STREAM CYCLE #{i:02d}] ---")
        # 1. Sensor generates observation
        raw_event = fleet.generate_next_event()

        # 2. Producer publishes to Kafka topic
        producer.send_event(raw_event, verbose=True)

        # 3. Consumer receives event from Kafka topic
        consumed_events = list(consumer.consume_events(max_messages=1, timeout=0.5, verbose=True))
        if consumed_events:
            ev = consumed_events[0]
            lat = ev['latitude']
            lon = ev['longitude']
            print(f"[STREAM] Validated Event ID: {ev['event_id'][:8]}... | Location: ({lat:.4f}, {lon:.4f})")

    producer.close()
    consumer.close()

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Streaming demonstration complete! {args.events} events produced and consumed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
