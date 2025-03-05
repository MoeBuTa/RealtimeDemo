import json
import os
import threading
import time
import datetime
from typing import Any, Dict, Optional

from loguru import logger
import websocket

from realtime.hardware.player import AudioPlayer
from realtime.hardware.recorder import AudioRecorder
from realtime.utils.config import HEADERS, REALTIME_URL
from realtime.utils.constants import CONVERSATION_DIR


class RealtimeClient:
    """Client for the OpenAI Realtime API with improved barge-in support, custom instructions and response recording"""

    def __init__(self, default_instructions: str = "Please assist the user."):
        self.ws = None
        self.player = AudioPlayer()
        self.recorder = AudioRecorder(
            threshold=0.02,  # Adjust based on your microphone sensitivity
            silence_duration=1.0,  # Stop recording after 1 second of silence
            min_speech_duration=0.5,  # Require at least 0.5 seconds of speech
            pre_buffer_duration=0.5  # Keep 0.5 seconds before speech detection
        )
        self.connected = False
        self.current_event_id = 0
        self.processing_input = False  # Flag to prevent overlapping processing
        self.assistant_speaking = False  # Flag to track if assistant is speaking
        self.current_response_id = None  # Track the current response ID
        self.expecting_audio = False  # Flag to track if we're expecting new audio
        self.first_audio_chunk = True  # Flag to track first audio chunk in response
        self.default_instructions = default_instructions  # Store default instructions

        # Response recording settings
        self.output_dir = CONVERSATION_DIR

        # Store current text response
        self.current_response_text = ""
        self.conversation_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def connect(self):
        """Connect to the Realtime API"""

        def on_open(ws):
            self.connected = True
            logger.info("Connected to OpenAI Realtime API")

        def on_message(ws, message):
            try:
                event = json.loads(message)
                event_type = event.get("type")

                if event_type == "input_audio_buffer.speech_started":
                    logger.success("Speech detected in input audio buffer")

                elif event_type == "input_audio_buffer.committed":
                    logger.success("Input audio buffer committed")

                elif event_type == "conversation.item.created":
                    logger.success(f"Conversation item created: {json.dumps(event.get('item', {}), indent=2)}")

                elif event_type == "response.created":
                    logger.success(f"Response created: {json.dumps(event.get('response', {}), indent=2)}")
                    self.current_response_id = event.get("response").get("id")
                    # Will expect audio for this response
                    self.expecting_audio = True
                    self.first_audio_chunk = True
                elif event_type == "rate_limits.updated":
                    logger.success(f"Rate limits updated: {json.dumps(event.get('rate_limits', {}), indent=2)}")

                elif event_type == "response.output_item.added":
                    logger.success(f"Output item added: {json.dumps(event.get('item', {}), indent=2)}")

                elif event_type == "response.content_part.added":
                    logger.success(f"Content part added: {json.dumps(event.get('part', {}), indent=2)}")

                elif event_type == "response.audio.delta":
                    logger.success(f"Audio delta updated: Response ID {event.get('response_id')}, Item ID {event.get('item_id')}")
                    # If this is the first audio chunk, force restart the player
                    if self.first_audio_chunk or self.expecting_audio:
                        logger.info("First audio delta received - force restarting player")
                        self.player.force_restart()
                        self.first_audio_chunk = False
                        self.expecting_audio = False
                    self.assistant_speaking = True  # Mark that assistant is speaking
                    # Get the audio data directly from the delta field
                    audio_data = event.get("delta", "")
                    logger.info(f"Processing audio delta of size: {len(audio_data)} bytes")
                    if audio_data:
                        self.player.add_audio(audio_data)
                    else:
                        logger.warning("Audio delta contained no audio data")

                elif event_type == "response.audio_transcript.delta":
                    logger.success(f"Audio transcript delta updated: {event.get('delta', {})}")
                    text = event.get("delta", {})
                    # Append to the current response text
                    self.current_response_text += text
                    # If recording is enabled, save the text incrementally
                    if self.output_dir and self.current_response_id:
                        self._save_response_text()

                elif event_type == "error":
                    error = event.get("error", {})
                    logger.error(f"Error: {error.get('message', 'Unknown error')}")

                elif event_type == "response.done":
                    logger.success("Response complete.")

                    # Final save of the response text
                    if self.output_dir and self.current_response_text:
                        self._save_response_text(final=True)

                    # Reset response text for next interaction
                    self.current_response_text = ""

                    # Reset flags when response is complete
                    self.processing_input = False
                    self.assistant_speaking = False
                    self.current_response_id = None
                    self.first_audio_chunk = True
                    self.expecting_audio = False

                    self.send_event("session.update", {
                        "session": {
                            "tools": [],
                            "temperature": 1.0
                        }
                    })

                elif event_type == "session.updated":
                    logger.info("Session configuration updated.")

                # For debugging other event types
                else:
                    logger.info(f"Event: {event_type}")

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

        # Set up recorder callback with barge-in support
        self.recorder.set_recording_finished_callback(self.process_recorded_audio)
        # Set up speech detection callback for barge-in detection
        self.recorder.set_speech_detected_callback(self.handle_barge_in)

    def disconnect(self):
        """Disconnect from the Realtime API"""
        if self.ws:
            self.ws.close()

        self.player.stop()
        self.recorder.stop_listening()

    def send_event(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """Send an event to the Realtime API"""
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

    def create_response(self, instructions: Optional[str] = None, modalities: list = None):
        """Create a response with custom instructions"""
        if not modalities:
            modalities = ["text", "audio"]  # Default to both text and audio

        # Use provided instructions or fall back to default
        instructions_text = instructions if instructions is not None else self.default_instructions

        logger.info(f"Creating response with instructions: {instructions_text}")

        # Create a response with custom instructions
        self.send_event("response.create", {
            "response": {
                "modalities": modalities,
                "instructions": instructions_text
            }
        })

        # Set flag to expect new audio
        self.expecting_audio = True
        self.first_audio_chunk = True

    def handle_barge_in(self):
        """Handle barge-in with complete player stopping"""
        if self.assistant_speaking:
            logger.info("BARGE-IN DETECTED: User started speaking while assistant was talking")

            # Cancel the current response if there is one
            if self.current_response_id:
                logger.info(f"Canceling response {self.current_response_id} due to barge-in")
                try:
                    self.send_event("response.cancel", {"response_id": self.current_response_id})
                except Exception as e:
                    logger.warning(f"Error canceling response: {e}")

            # Force kill the player completely
            self.player.stop()

            # Wait a small amount of time to ensure clean shutdown
            time.sleep(0.1)

            # Start a new player instance
            self.player.start()

            # Reset assistant speaking flag
            self.assistant_speaking = False

            # Show visual indicator of barge-in
            print("\n[User interrupted - listening...]")

    def process_recorded_audio(self, custom_instructions: Optional[str] = None):
        """Process the recorded audio data when recording finished"""
        # Avoid processing multiple inputs simultaneously
        if self.processing_input:
            logger.warning("Already processing input, skipping...")
            return

        self.processing_input = True

        try:
            audio_data = self.recorder.get_base64_audio()
            if not audio_data:
                logger.warning("No audio recorded, skipping.")
                self.processing_input = False
                return

            logger.info(f"Processing recorded audio ({len(audio_data)} bytes)...")

            # Completely stop the player for new input
            if self.assistant_speaking:
                # Stop the player and wait for it to fully stop
                self.player.stop()
                time.sleep(0.1)
                self.player.start()
                self.assistant_speaking = False

            # Append the audio to the input buffer
            self.send_event("input_audio_buffer.append", {"audio": audio_data})

            # Commit the audio buffer
            self.send_event("input_audio_buffer.commit")

            # Create a response with custom instructions if provided
            self.create_response(instructions=custom_instructions)

            # Clear the recording for the next interaction
            self.recorder.clear_recording()

        except Exception as e:
            logger.exception(f"Error processing recorded audio: {e}")
            self.processing_input = False

    def _save_response_text(self, final: bool = False):
        """Save the current response text to a file"""
        if not self.output_dir or not self.current_response_text:
            return

        # Create a filename based on the response ID and whether this is the final version
        filename_prefix = f"response_{self.conversation_id}_{self.current_response_id}"
        filename = f"{filename_prefix}_final.txt" if final else f"{filename_prefix}_partial.txt"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.current_response_text)

            if final:
                logger.info(f"Saved final response to {filepath}")

                # Remove any partial files for this response
                partial_path = os.path.join(self.output_dir, f"{filename_prefix}_partial.txt")
                if os.path.exists(partial_path):
                    os.remove(partial_path)
        except Exception as e:
            logger.error(f"Error saving response text: {e}")

    def run_conversation(self, instructions: Optional[str] = None):
        """Run an interactive conversation with automatic voice detection and optional custom instructions"""
        try:
            if instructions:
                print(f"\nStarting conversation with instructions: '{instructions}'")
            else:
                print(f"\nStarting conversation with default instructions: '{self.default_instructions}'")

            if self.output_dir:
                print(f"Saving responses to: {self.output_dir}")

            print("You can interrupt the assistant by speaking while it's talking.")
            print("(Press Ctrl+C to exit)\n")

            # Store instructions for this conversation
            conversation_instructions = instructions if instructions else self.default_instructions

            # Override the recorder's callback to include these instructions
            def process_with_instructions():
                self.process_recorded_audio(conversation_instructions)

            # Update the callback
            self.recorder.set_recording_finished_callback(process_with_instructions)

            # Start listening for voice
            self.recorder.start_listening()

            # Keep the main thread alive
            while self.connected:
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Exiting...")
        finally:
            self.disconnect()
