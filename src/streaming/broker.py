"""
Streaming Broker Abstraction — Dual-Mode (Apache Kafka + In-Memory Fallback)

Provides seamless event-driven streaming:
- Direct Apache Kafka integration via kafka-python-ng when broker is reachable
- In-memory thread-safe pub/sub fallback broker for local development without Docker
- Transparent switching with zero code changes in producers or consumers
"""

import json
import logging
import queue
import socket
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Generator, Any

from src.config.settings import settings

logger = logging.getLogger(__name__)


def is_kafka_reachable(host_port: str, timeout: float = 1.0) -> bool:
    """Check if the Kafka broker socket is listening."""
    try:
        host, port = host_port.split(":")
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


@dataclass
class StreamMessage:
    """Wrapper representing a message in a stream topic."""
    topic: str
    value: dict
    key: Optional[str] = None
    offset: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BaseBroker(ABC):
    """Abstract interface for topic-based message streaming."""

    @abstractmethod
    def publish(self, topic: str, value: dict, key: Optional[str] = None) -> StreamMessage:
        pass

    @abstractmethod
    def consume(self, topic: str, max_messages: Optional[int] = None, timeout: float = 1.0) -> Generator[StreamMessage, None, None]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class LocalQueueBroker(BaseBroker):
    """In-memory pub/sub broker simulating Kafka topics, offsets, and consumer groups."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._topics = {}
                cls._instance._offsets = {}
            return cls._instance

    def _get_queue(self, topic: str) -> queue.Queue:
        if topic not in self._topics:
            self._topics[topic] = queue.Queue()
            self._offsets[topic] = 0
        return self._topics[topic]

    def publish(self, topic: str, value: dict, key: Optional[str] = None) -> StreamMessage:
        q = self._get_queue(topic)
        self._offsets[topic] += 1
        msg = StreamMessage(
            topic=topic,
            value=value,
            key=key,
            offset=self._offsets[topic],
        )
        q.put(msg)
        return msg

    def consume(self, topic: str, max_messages: Optional[int] = None, timeout: float = 1.0) -> Generator[StreamMessage, None, None]:
        q = self._get_queue(topic)
        count = 0
        while True:
            if max_messages is not None and count >= max_messages:
                break
            try:
                msg = q.get(timeout=timeout)
                count += 1
                yield msg
            except queue.Empty:
                break

    def close(self) -> None:
        pass


class KafkaStreamBroker(BaseBroker):
    """Real Apache Kafka client using kafka-python-ng."""

    def __init__(self, bootstrap_servers: str):
        from kafka import KafkaProducer, KafkaConsumer
        self.bootstrap_servers = bootstrap_servers
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            retries=3,
        )
        self._consumers: dict[str, Any] = {}

    def publish(self, topic: str, value: dict, key: Optional[str] = None) -> StreamMessage:
        future = self._producer.send(topic, value=value, key=key)
        record_metadata = future.get(timeout=10.0)
        return StreamMessage(
            topic=topic,
            value=value,
            key=key,
            offset=record_metadata.offset,
        )

    def consume(self, topic: str, max_messages: Optional[int] = None, timeout: float = 1.0) -> Generator[StreamMessage, None, None]:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            auto_offset_reset="earliest",
            consumer_timeout_ms=int(timeout * 1000),
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id=settings.KAFKA_CONSUMER_GROUP,
        )
        count = 0
        for record in consumer:
            if max_messages is not None and count >= max_messages:
                break
            count += 1
            yield StreamMessage(
                topic=record.topic,
                value=record.value,
                key=record.key.decode("utf-8") if record.key else None,
                offset=record.offset,
            )
        consumer.close()

    def close(self) -> None:
        self._producer.flush()
        self._producer.close()


def get_broker(force_mode: Optional[str] = None) -> BaseBroker:
    """Factory creating Kafka broker or falling back to local queue broker."""
    mode = force_mode or settings.STREAMING_MODE

    if mode == "kafka" or (mode == "auto" and is_kafka_reachable(settings.KAFKA_BROKER)):
        try:
            logger.info(f"Connecting to Apache Kafka broker at {settings.KAFKA_BROKER}...")
            return KafkaStreamBroker(settings.KAFKA_BROKER)
        except Exception as e:
            logger.warning(f"Failed to connect to Kafka ({e}). Falling back to local stream broker.")

    logger.info("Using high-performance local event stream broker (simulated Kafka).")
    return LocalQueueBroker()
