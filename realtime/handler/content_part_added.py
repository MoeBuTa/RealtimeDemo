# realtime/handler/content_part_added.py
import json
from typing import Any, Dict
from loguru import logger

from realtime.handler.base import EventHandler


class ContentPartAddedEventHandler(EventHandler):
    """Handler for content part added events"""

    def handle(self, event: Dict[str, Any], **kwargs) -> None:
        """
        Handle content part added event

        Args:
            event: The event data
            **kwargs: Additional context from the client
        """
        logger.success(f"Content part added: {json.dumps(event.get('part', {}), indent=2)}")