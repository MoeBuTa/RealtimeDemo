# realtime/handler/rate_limits_updated.py
import json
from typing import Any, Dict
from loguru import logger

from realtime.handler.base import EventHandler


class RateLimitsUpdatedEventHandler(EventHandler):
    """Handler for rate limits updated events"""

    def handle(self, event: Dict[str, Any], **kwargs) -> None:
        """
        Handle rate limits updated event

        Args:
            event: The event data
            **kwargs: Additional context from the client
        """
        logger.success(f"Rate limits updated: {json.dumps(event.get('rate_limits', {}), indent=2)}")