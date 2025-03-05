# realtime/service/audio_processor.py
import time
from typing import Dict, Any, Optional, Callable, List
from loguru import logger

from realtime.service.response_creator import ResponseCreator


class AudioProcessor:
    """Service for processing recorded audio"""

    @staticmethod
    def process(send_event_callback: Callable,
                recorder: Any,
                player: Any,
                state_updater: Callable,
                default_instructions: str,
                client_state: Dict[str, Any],
                **kwargs) -> None:
        """
        Process the recorded audio data when recording finished

        Args:
            send_event_callback: Function to send events
            recorder: Audio recorder instance
            player: Audio player instance
            state_updater: Function to update client state
            default_instructions: Default instructions
            client_state: Current client state
            **kwargs: Additional parameters for response creation including:
                - instructions: Custom instructions for the assistant
                - voice: Voice to use for audio responses
                - tools: List of tools available to the assistant
                - tool_choice: Tool selection strategy
                - temperature: Temperature setting for response generation
                - max_output_tokens: Maximum number of tokens to generate
        """
        # Avoid processing multiple inputs simultaneously
        if client_state.get('processing_input', False):
            logger.warning("Already processing input, skipping...")
            return

        state_updater({'processing_input': True})

        try:
            audio_data = recorder.get_base64_audio()
            if not audio_data:
                logger.warning("No audio recorded, skipping.")
                state_updater({'processing_input': False})
                return

            logger.info(f"Processing recorded audio ({len(audio_data)} bytes)...")

            # Completely stop the player for new input
            if client_state.get('assistant_speaking', False):
                # Stop the player and wait for it to fully stop
                player.stop()
                time.sleep(0.1)
                player.start()
                state_updater({'assistant_speaking': False})

            # Append the audio to the input buffer
            send_event_callback("input_audio_buffer.append", {"audio": audio_data})

            # Commit the audio buffer
            send_event_callback("input_audio_buffer.commit")

            # Prepare parameters for response creation
            response_params = {
                'send_event_callback': send_event_callback,
                'default_instructions': default_instructions,
                'state_updater': state_updater,
            }

            # Add any additional parameters provided
            response_params.update(kwargs)

            # Create a response with all provided parameters
            ResponseCreator.create(**response_params)

            # Clear the recording for the next interaction
            recorder.clear_recording()

        except Exception as e:
            logger.exception(f"Error processing recorded audio: {e}")
            state_updater({'processing_input': False})