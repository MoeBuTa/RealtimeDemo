# realtime/handler/conversation_item_created.py
import json
from typing import Any, Dict
from loguru import logger

from realtime.handler.base import EventHandler


class ConversationItemCreatedEventHandler(EventHandler):
    """Handler for conversation item created events"""

    def handle(self, event: Dict[str, Any], **kwargs) -> None:
        """
        Handle conversation item created event

        Args:
            event: The event data
            **kwargs: Additional context from the client
        """
        logger.success(f"Conversation item created: {json.dumps(event.get('item', {}), indent=2)}")