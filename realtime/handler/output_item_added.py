# realtime/handler/output_item_added.py
import json
from typing import Any, Dict
from loguru import logger

from realtime.handler.base import EventHandler


class OutputItemAddedEventHandler(EventHandler):
    """Handler for output item added events"""

    def handle(self, event: Dict[str, Any], **kwargs) -> None:
        """
        Handle output item added event

        Args:
            event: The event data
            **kwargs: Additional context from the client
        """
        logger.success(f"Output item added: {json.dumps(event.get('item', {}), indent=2)}")