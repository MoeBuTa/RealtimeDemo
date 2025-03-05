# realtime/llm/client.py
import json
import os
import threading
import time
import datetime
from typing import Any, Dict, Optional, List

from loguru import logger
import websocket

from realtime.hardware.player import AudioPlayer
from realtime.hardware.recorder import AudioRecorder
from realtime.utils.config import HEADERS, REALTIME_URL
from realtime.utils.constants import CONVERSATION_DIR
from realtime.handler.factory import EventHandlerFactory
from realtime.service.response_creator import ResponseCreator
from realtime.service.audio_processor import AudioProcessor
from realtime.service.bargein_handler import BargeinHandler


class RealtimeClient:
    """Client for the OpenAI Realtime API with improved architecture to avoid circular imports"""

    def __init__(self, default_instructions: str = "Please assist the user."):
        # Initialize hardware components
        self.ws = None
        self.player = AudioPlayer()
        self.recorder = AudioRecorder(
            threshold=0.02,  # Adjust based on your microphone sensitivity
            silence_duration=1.0,  # Stop recording after 1 second of silence
            min_speech_duration=0.5,  # Require at least 0.5 seconds of speech
            pre_buffer_duration=0.5  # Keep 0.5 seconds before speech detection
        )

        # Initialize connection state
        self.connected = False
        self.current_event_id = 0

        # Initialize default instructions
        self.default_instructions = default_instructions

        # Initialize response recording settings
        self.output_dir = CONVERSATION_DIR
        self.conversation_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize client state
        self.state = {
            'processing_input': False,
            'assistant_speaking': False,
            'current_response_id': None,
            'expecting_audio': False,
            'first_audio_chunk': True,
            'current_response_text': ""
        }

        # Initialize handler factory
        EventHandlerFactory.initialize_default_handlers()

    def update_state(self, state_updates: Dict[str, Any]) -> None:
        """
        Update the client state

        Args:
            state_updates: Dictionary of state values to update
        """
        self.state.update(state_updates)

    def connect(self):
        """Connect to the Realtime API"""

        def on_open(ws):
            self.connected = True
            logger.info("Connected to OpenAI Realtime API")

        def on_message(ws, message):
            try:
                event = json.loads(message)
                event_type = event.get("type")

                # Get the appropriate handler for this event type
                handler = EventHandlerFactory.get_handler(event_type)

                # Prepare common kwargs for all handlers
                kwargs = {
                    'player': self.player,
                    'recorder': self.recorder,
                    'state_updater': self.update_state,
                    'client_state': self.state,
                    'output_dir': self.output_dir,
                    'save_response_callback': self._save_response_text,
                    'send_event_callback': self.send_event,
                    'logger': logger.info,
                    'default_instructions': self.default_instructions
                }

                # Handle the event
                handler.handle(event, **kwargs)

            except Exception as e:
                logger.exception(f"Error processing message: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            self.connected = False

        def on_close(ws, close_status_code, close_reason):
            logger.info(f"Connection closed: {close_status_code} - {close_reason}")
            self.connected = False

        # Initialize WebSocket connection
        self.ws = websocket.WebSocketApp(
            REALTIME_URL,
            header=HEADERS,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # Start the WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        # Wait for connection to establish
        timeout = 10
        start_time = time.time()
        while not self.connected and time.time() - start_time < timeout:
            time.sleep(0.1)

        if not self.connected:
            raise Exception("Connection timeout")

        # Start the audio player
        self.player.start()

        # Set up recorder callbacks
        self.recorder.set_recording_finished_callback(self.process_recorded_audio)
        self.recorder.set_speech_detected_callback(self.handle_barge_in)

    def disconnect(self):
        """Disconnect from the Realtime API"""
        if self.ws:
            self.ws.close()

        self.player.stop()
        self.recorder.stop_listening()

    def send_event(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """
        Send an event to the Realtime API

        Args:
            event_type: The type of event to send
            data: Additional data for the event
        """
        if not self.ws or not self.connected:
            logger.warning("Not connected to the API")
            return

        self.current_event_id += 1
        event_id = f"evt_{self.current_event_id}"

        event = {"type": event_type, "event_id": event_id}
        if data:
            event.update(data)

        logger.debug(f"Sending event: {json.dumps(event, indent=2)}")
        self.ws.send(json.dumps(event))

    def _save_response_text(self, final: bool = False):
        """
        Save the current response text to a file

        Args:
            final: Whether this is the final save for the response
        """
        if not self.output_dir or not self.state.get('current_response_text'):
            return

        current_response_id = self.state.get('current_response_id')
        if not current_response_id:
            return

        # Create a filename based on the response ID and whether this is the final version
        filename_prefix = f"response_{self.conversation_id}_{current_response_id}"
        filename = f"{filename_prefix}_final.txt" if final else f"{filename_prefix}_partial.txt"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.state.get('current_response_text', ''))

            if final:
                logger.info(f"Saved final response to {filepath}")

                # Remove any partial files for this response
                partial_path = os.path.join(self.output_dir, f"{filename_prefix}_partial.txt")
                if os.path.exists(partial_path):
                    os.remove(partial_path)
        except Exception as e:
            logger.error(f"Error saving response text: {e}")

    def create_response(self, instructions: Optional[str] = None,
                        modalities: List[str] = None,
                        voice: str = "sage",
                        output_audio_format: str = "pcm16",
                        tools: List[Dict] = None,
                        tool_choice: str = "auto",
                        temperature: float = 0.8,
                        max_output_tokens: int = 1024):
        """
        Create a response with customizable parameters

        Args:
            instructions: Custom instructions for the assistant
            modalities: Response modalities
            voice: Voice to use for audio responses
            output_audio_format: Format for audio output
            tools: List of tools available to the assistant
            tool_choice: Tool selection strategy
            temperature: Temperature setting for response generation
            max_output_tokens: Maximum number of tokens to generate
        """
        ResponseCreator.create(
            send_event_callback=self.send_event,
            default_instructions=self.default_instructions,
            state_updater=self.update_state,
            instructions=instructions,
            modalities=modalities,
            voice=voice,
            output_audio_format=output_audio_format,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )

    def handle_barge_in(self):
        """Handle barge-in with complete player stopping"""
        BargeinHandler.handle(
            send_event_callback=self.send_event,
            player=self.player,
            state_updater=self.update_state,
            client_state=self.state
        )

    def process_recorded_audio(self, **kwargs):
        """
        Process the recorded audio data when recording finished

        Args:
            **kwargs: Parameters for response creation including instructions, voice, etc.
        """
        # Create processor kwargs from client properties
        processor_kwargs = {
            'send_event_callback': self.send_event,
            'recorder': self.recorder,
            'player': self.player,
            'state_updater': self.update_state,
            'default_instructions': self.default_instructions,
            'client_state': self.state,
        }

        # Add any additional parameters from the function call
        processor_kwargs.update(kwargs)

        # Process the audio with all parameters
        AudioProcessor.process(**processor_kwargs)

    def run_conversation(self, instructions: Optional[str] = None,
                         voice: str = "sage",
                         tools: List[Dict] = None,
                         temperature: float = 0.8,
                         max_output_tokens: int = 1024):
        """
        Run an interactive conversation with automatic voice detection

        Args:
            instructions: Custom instructions for the assistant
            voice: Voice to use for audio responses
            tools: List of tools available to the assistant
            temperature: Temperature setting for response generation
            max_output_tokens: Maximum number of tokens to generate
        """
        try:
            if instructions:
                print(f"\nStarting conversation with instructions: '{instructions}'")
            else:
                print(f"\nStarting conversation with default instructions: '{self.default_instructions}'")

            if self.output_dir:
                print(f"Saving responses to: {self.output_dir}")

            print("You can interrupt the assistant by speaking while it's talking.")
            print("(Press Ctrl+C to exit)\n")

            # Store conversation configuration
            conversation_config = {}

            # Only add parameters that are not None
            if instructions is not None:
                conversation_config['instructions'] = instructions
            if voice:
                conversation_config['voice'] = voice
            if tools:
                conversation_config['tools'] = tools
            if temperature:
                conversation_config['temperature'] = temperature
            if max_output_tokens:
                conversation_config['max_output_tokens'] = max_output_tokens

            # Create a callback that will use these parameters
            def process_with_config():
                self.process_recorded_audio(**conversation_config)

            # Update the callback
            self.recorder.set_recording_finished_callback(process_with_config)

            # Start listening for voice
            self.recorder.start_listening()

            # Keep the main thread alive
            while self.connected:
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Exiting...")
        finally:
            self.disconnect()
