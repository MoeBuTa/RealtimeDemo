# realtime/handler/response_created.py
import json
from typing import Any, Dict, Callable
from loguru import logger

from realtime.handler.base import EventHandler


class ResponseCreatedEventHandler(EventHandler):
    """Handler for response created events"""

    def handle(self, event: Dict[str, Any], **kwargs) -> None:
        """
        Handle response created event

        Args:
            event: The event data
            **kwargs: Additional context including state_updater callback
        """
        logger.success(f"Response created: {json.dumps(event.get('response', {}), indent=2)}")

        # Extract the response ID
        response_id = event.get("response", {}).get("id")

        # Use the state updater function to update client state without direct reference
        state_updater = kwargs.get('state_updater')
        if state_updater and response_id:
            state_updater({
                'current_response_id': response_id,
                'expecting_audio': True,
                'first_audio_chunk': True
            })