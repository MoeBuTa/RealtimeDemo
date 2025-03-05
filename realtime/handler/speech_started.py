# realtime/handler/speech_started.py
from typing import Any, Dict
from loguru import logger

from realtime.handler.base import EventHandler


class SpeechStartedEventHandler(EventHandler):
    """Handler for speech started events"""

    def handle(self, event: Dict[str, Any], **kwargs) -> None:
        """
        Handle speech started event

        Args:
            event: The event data
            **kwargs: Additional context from the client
        """
        logger.success("Speech detected in input audio buffer")