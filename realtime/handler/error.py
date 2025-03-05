# realtime/handler/error.py
from typing import Any, Dict
from loguru import logger

from realtime.handler.base import EventHandler


class ErrorEventHandler(EventHandler):
    """Handler for error events"""

    def handle(self, event: Dict[str, Any], **kwargs) -> None:
        """
        Handle error event

        Args:
            event: The event data
            **kwargs: Additional context from the client
        """
        error = event.get("error", {})
        logger.error(f"Error: {error.get('message', 'Unknown error')}")