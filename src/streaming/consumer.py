"""
Noise Event Consumer — Consumes, deserializes, and processes stream events

Subscribes to:
- Topic: urban-noise-events

Follows Master Prompt Section 18 logging conventions:
[CONSUMER] Event received
[ML] Prediction = ...
"""

import logging
from typing import Optional, Callable, Generator
from src.config.settings import settings
from src.streaming.broker import BaseBroker, StreamMessage, get_broker

logger = logging.getLogger(__name__)


class NoiseEventConsumer:
    """Consumes and processes streaming events from Kafka."""

    def __init__(
        self,
        broker: Optional[BaseBroker] = None,
        topic: Optional[str] = None,
        group_id: Optional[str] = None,
    ):
        self.broker = broker or get_broker()
        self.topic = topic or settings.KAFKA_TOPIC_EVENTS
        self.group_id = group_id or settings.KAFKA_CONSUMER_GROUP

    def consume_events(
        self,
        max_messages: Optional[int] = None,
        timeout: float = 1.0,
        callback: Optional[Callable[[dict], None]] = None,
        verbose: bool = True,
    ) -> Generator[dict, None, None]:
        """Poll and stream events from topic.

        Args:
            max_messages: Maximum messages to consume before returning. If None, runs indefinitely.
            timeout: Timeout in seconds for empty queue.
            callback: Optional callback invoked with the deserialized payload dict.
            verbose: Prints presentation logs.

        Yields:
            Deserialized event dictionary.
        """
        for msg in self.broker.consume(topic=self.topic, max_messages=max_messages, timeout=timeout):
            event = msg.value

            if verbose:
                print(f"[CONSUMER] Event received from '{msg.topic}' (offset: {msg.offset})")
                print(
                    f"[ML] Prediction = {event.get('sound_class', 'unknown')} "
                    f"(conf: {event.get('confidence', 0)*100:0.1f}%) | "
                    f"Noise: {event.get('calibrated_db', 0):0.1f} dB"
                )

            if callback is not None:
                callback(event)

            yield event

    def close(self) -> None:
        self.broker.close()
