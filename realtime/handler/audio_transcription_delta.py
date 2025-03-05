# realtime/handler/audio_transcript_delta.py
from typing import Any, Dict
from loguru import logger

from realtime.handler.base import EventHandler


class AudioTranscriptDeltaEventHandler(EventHandler):
    """Handler for audio transcript delta events"""

    def handle(self, event: Dict[str, Any], **kwargs) -> None:
        """
        Handle audio transcript delta event

        Args:
            event: The event data
            **kwargs: Additional context including:
                - state_updater: function to update client state
                - client_state: current client state dict
                - save_response_callback: function to save response text
        """
        logger.success(f"Audio transcript delta updated: {event.get('delta', {})}")

        # Get the text delta
        text = event.get("delta", "")

        # Get dependencies from kwargs
        state_updater = kwargs.get('state_updater')
        client_state = kwargs.get('client_state', {})
        save_response_callback = kwargs.get('save_response_callback')

        if not state_updater:
            logger.error("Required state_updater dependency missing in AudioTranscriptDeltaEventHandler")
            return

        # Get current response text
        current_response_text = client_state.get('current_response_text', '')

        # Update response text with new delta
        updated_text = current_response_text + text
        state_updater({'current_response_text': updated_text})

        # If recording is enabled, save the text incrementally
        output_dir = kwargs.get('output_dir')
        current_response_id = client_state.get('current_response_id')

        if output_dir and current_response_id and save_response_callback:
            save_response_callback(final=False)