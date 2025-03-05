from typing import Any, Dict

from loguru import logger

from realtime.handler.base import EventHandler
from realtime.llm.client import RealtimeClient


class DefaultEventHandler(EventHandler):
    """Default handler for unknown event types"""

    def handle(self, event: Dict[str, Any], client: 'RealtimeClient') -> None:
        event_type = event.get("type")
        logger.info(f"Event: {event_type}")