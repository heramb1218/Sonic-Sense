"""
Noise Event Producer — Serializes and publishes acoustic events to Kafka topics

Emits structured events to:
- Topic: urban-noise-events

Follows Master Prompt Section 18 logging conventions.
"""

import logging
from typing import Optional
from src.config.settings import settings
from src.streaming.broker import BaseBroker, StreamMessage, get_broker

logger = logging.getLogger(__name__)


class NoiseEventProducer:
    """Publishes acoustic sensor observation events to Kafka topics."""

    def __init__(self, broker: Optional[BaseBroker] = None, topic: Optional[str] = None):
        self.broker = broker or get_broker()
        self.topic = topic or settings.KAFKA_TOPIC_EVENTS

    def send_event(self, event: dict, verbose: bool = True) -> StreamMessage:
        """Publish an acoustic event to the message topic.

        Args:
            event: Standard observation event dictionary.
            verbose: If True, prints presentation logs matching Master Prompt Section 18.

        Returns:
            StreamMessage object containing topic, offset, and payload.
        """
        device_id = event.get("device_id", "unknown_sensor")
        key = device_id

        msg = self.broker.publish(topic=self.topic, value=event, key=key)

        if verbose:
            print(f"[SENSOR] {device_id} generated observation")
            print(f"[KAFKA] Event published to '{self.topic}' (offset: {msg.offset})")

        return msg

    def close(self) -> None:
        self.broker.close()
