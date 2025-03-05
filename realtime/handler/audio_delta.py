# realtime/handler/audio_delta.py
from typing import Any, Dict, Callable
from loguru import logger

from realtime.handler.base import EventHandler


class AudioDeltaEventHandler(EventHandler):
    """Handler for audio delta events"""

    def handle(self, event: Dict[str, Any], **kwargs) -> None:
        """
        Handle audio delta event

        Args:
            event: The event data
            **kwargs: Additional context including:
                - player: AudioPlayer instance
                - state_updater: function to update client state
                - client_state: current client state dict
        """
        logger.success(f"Audio delta updated: Response ID {event.get('response_id')}, Item ID {event.get('item_id')}")

        # Get dependencies from kwargs
        player = kwargs.get('player')
        state_updater = kwargs.get('state_updater')
        client_state = kwargs.get('client_state', {})

        if not player or not state_updater:
            logger.error("Required dependencies missing in AudioDeltaEventHandler")
            return

        # Check if this is the first audio chunk or we're expecting audio
        if client_state.get('first_audio_chunk', False) or client_state.get('expecting_audio', False):
            logger.info("First audio delta received - force restarting player")
            player.force_restart()

            # Update state
            state_updater({
                'first_audio_chunk': False,
                'expecting_audio': False,
                'assistant_speaking': True
            })
        elif not client_state.get('assistant_speaking', False):
            # Update that assistant is speaking
            state_updater({'assistant_speaking': True})

        # Get the audio data directly from the delta field
        audio_data = event.get("delta", "")
        logger.info(f"Processing audio delta of size: {len(audio_data)} bytes")

        # Play the audio if we have data
        if audio_data and player:
            player.add_audio(audio_data)
        else:
            logger.warning("Audio delta contained no audio data")