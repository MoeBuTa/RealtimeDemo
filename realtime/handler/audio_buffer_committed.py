# realtime/handler/audio_buffer_committed.py
from typing import Any, Dict
from loguru import logger

from realtime.handler.base import EventHandler


class AudioBufferCommittedEventHandler(EventHandler):
    """Handler for audio buffer committed events"""

    def handle(self, event: Dict[str, Any], **kwargs) -> None:
        """
        Handle audio buffer committed event

        Args:
            event: The event data
            **kwargs: Additional context from the client
        """
        logger.success("Input audio buffer committed")