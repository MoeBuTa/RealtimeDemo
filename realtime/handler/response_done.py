# realtime/handler/response_done.py
from typing import Any, Dict, Callable, Optional
from loguru import logger

from realtime.handler.base import EventHandler


class ResponseDoneEventHandler(EventHandler):
    """Handler for response done events"""

    def handle(self, event: Dict[str, Any], **kwargs) -> None:
        """
        Handle response done event

        Args:
            event: The event data
            **kwargs: Additional context including:
                - output_dir: directory for saving responses
                - current_response_text: text of current response
                - current_response_id: ID of current response
                - save_response_callback: function to save response text
                - state_updater: function to update client state
                - send_event_callback: function to send events
        """
        logger.success("Response complete.")

        # Extract dependencies
        output_dir = kwargs.get('output_dir')
        current_response_text = kwargs.get('client_state', {}).get('current_response_text', '')
        current_response_id = kwargs.get('client_state', {}).get('current_response_id')
        save_response_callback = kwargs.get('save_response_callback')
        state_updater = kwargs.get('state_updater')
        send_event_callback = kwargs.get('send_event_callback')

        # Final save of the response text
        if output_dir and current_response_text and current_response_id and save_response_callback:
            save_response_callback(final=True)

        # Reset state when response is complete
        if state_updater:
            state_updater({
                'current_response_text': '',
                'processing_input': False,
                'assistant_speaking': False,
                'current_response_id': None,
                'first_audio_chunk': True,
                'expecting_audio': False
            })

        # Reset session for next interaction
        if send_event_callback:
            send_event_callback("session.update", {
                "session": {
                    "tools": [],
                    "temperature": 1.0
                }
            })