"""
Unit Tests for Phase 5 — Kafka Streaming and Event Broker

Tests verify:
- Message publication and FIFO consumption on topics
- Producer payload serialization and offset assignment
- Consumer event reception, deserialization, and callback triggering
- Fallback mechanics when Kafka daemon is offline
- Broker socket reachability detection

Usage:
    pytest tests/test_streaming.py -v
"""

import sys
import pytest

# Add project root to path
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.streaming.broker import (
    LocalQueueBroker,
    get_broker,
    is_kafka_reachable,
    StreamMessage,
)
from src.streaming.producer import NoiseEventProducer
from src.streaming.consumer import NoiseEventConsumer


class TestStreamingBroker:
    """Tests for the message stream broker."""

    def test_local_broker_publish_and_consume(self):
        broker = LocalQueueBroker()
        topic = "test_acoustic_topic"
        payload = {"event_id": "test_001", "noise_level_db": 68.5}

        msg = broker.publish(topic=topic, value=payload, key="test_sensor")
        assert isinstance(msg, StreamMessage)
        assert msg.topic == topic
        assert msg.value == payload
        assert msg.offset >= 1

        consumed = list(broker.consume(topic=topic, max_messages=1, timeout=0.2))
        assert len(consumed) == 1
        assert consumed[0].value == payload

    def test_broker_fallback_creation(self):
        broker = get_broker(force_mode="local")
        assert isinstance(broker, LocalQueueBroker)

    def test_kafka_socket_unreachable_on_dummy_port(self):
        # Port 19999 should not have a listening Kafka service
        assert not is_kafka_reachable("127.0.0.1:19999", timeout=0.1)


class TestProducerAndConsumer:
    """Tests for NoiseEventProducer and NoiseEventConsumer."""

    def test_producer_publishes_event(self):
        broker = LocalQueueBroker()
        producer = NoiseEventProducer(broker=broker, topic="test_producer_topic")
        event = {
            "event_id": "ev_12345",
            "device_id": "sensor_007",
            "sound_class": "siren",
            "calibrated_db": 82.0,
            "confidence": 0.88,
        }

        msg = producer.send_event(event, verbose=False)
        assert msg.offset >= 1
        assert msg.value["device_id"] == "sensor_007"

    def test_consumer_receives_in_fifo_order(self):
        broker = LocalQueueBroker()
        topic = "test_fifo_topic"
        producer = NoiseEventProducer(broker=broker, topic=topic)
        consumer = NoiseEventConsumer(broker=broker, topic=topic)

        # Send 3 events
        for i in range(3):
            producer.send_event({"order": i, "device_id": f"s_{i}"}, verbose=False)

        # Consume 3 events
        received = list(consumer.consume_events(max_messages=3, timeout=0.2, verbose=False))
        assert len(received) == 3
        assert [r["order"] for r in received] == [0, 1, 2]

    def test_consumer_triggers_callback(self):
        broker = LocalQueueBroker()
        topic = "test_callback_topic"
        producer = NoiseEventProducer(broker=broker, topic=topic)
        consumer = NoiseEventConsumer(broker=broker, topic=topic)

        producer.send_event({"device_id": "sensor_cb", "sound_class": "drilling"}, verbose=False)

        captured = []
        list(consumer.consume_events(max_messages=1, timeout=0.2, callback=lambda ev: captured.append(ev), verbose=False))

        assert len(captured) == 1
        assert captured[0]["sound_class"] == "drilling"
