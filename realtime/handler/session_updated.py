# realtime/handler/session_updated.py
from typing import Any, Dict
from loguru import logger

from realtime.handler.base import EventHandler


class SessionUpdatedEventHandler(EventHandler):
    """Handler for session updated events"""

    def handle(self, event: Dict[str, Any], **kwargs) -> None:
        """
        Handle session updated event

        Args:
            event: The event data
            **kwargs: Additional context from the client
        """
        logger.info("Session configuration updated.")