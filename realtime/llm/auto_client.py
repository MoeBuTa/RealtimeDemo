import json
import threading
import time
from typing import Dict, Any

import websocket

from realtime.hardware.auto_player import AutomaticAudioPlayer
from realtime.hardware.auto_recorder import AutomaticAudioRecorder
from realtime.utils.config import REALTIME_URL, HEADERS
from loguru import logger


class AutomaticRealtimeClient:
    """Client for the OpenAI Realtime API with improved barge-in support"""

    def __init__(self):
        self.ws = None
        self.player = AutomaticAudioPlayer()
        self.recorder = AutomaticAudioRecorder(
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

    def connect(self):
        """Connect to the Realtime API"""

        def on_open(ws):
            self.connected = True
            logger.info("Connected to OpenAI Realtime API")

        def on_message(ws, message):
            try:
                event = json.loads(message)
                event_type = event.get("type")

                # Track response ID if present
                if "response_id" in event:
                    self.current_response_id = event.get("response_id")
                    # When a new response is created, set flag to expect audio
                    if event_type == "response.created":
                        self.expecting_audio = True
                        self.first_audio_chunk = True
                        logger.info("Expecting new audio response")

                # Handle different event types
                if event_type == "audio.content":
                    logger.info("Received audio content event")

                    # If this is the first audio chunk, force restart the player
                    if self.first_audio_chunk or self.expecting_audio:
                        logger.info("First audio chunk received - force restarting player")
                        self.player.force_restart()
                        self.first_audio_chunk = False
                        self.expecting_audio = False

                    self.assistant_speaking = True  # Mark that assistant is speaking
                    audio_content = event.get("audio_content", {})
                    audio_data = audio_content.get("audio", "")
                    logger.info(f"Processing audio content of size: {len(audio_data)} bytes")
                    self.player.add_audio(audio_data)

                elif event_type == "response.audio.delta":
                    logger.info("Received audio delta event")

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

                elif event_type == "text.content":
                    text_content = event.get("text_content", {})
                    logger.info(f"Assistant: {text_content.get('text', '')}")

                elif event_type == "error":
                    error = event.get("error", {})
                    logger.error(f"Error: {error.get('message', 'Unknown error')}")

                elif event_type == "response.done":
                    logger.info("Response complete.")
                    # Reset flags when response is complete
                    self.processing_input = False
                    self.assistant_speaking = False
                    self.current_response_id = None
                    self.first_audio_chunk = True
                    self.expecting_audio = False

                elif event_type == "response.created":
                    logger.info(f"Response created with ID: {self.current_response_id}")
                    # Will expect audio for this response
                    self.expecting_audio = True
                    self.first_audio_chunk = True

                elif event_type == "session.created":
                    logger.info(f"Session created: {json.dumps(event.get('session', {}), indent=2)}")

                    # Now that we have the session created, we can try to update it
                    logger.info("Configuring session for audio...")
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

    def process_recorded_audio(self):
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

            # Create a response
            self.send_event("response.create")

            # Set flag to expect new audio
            self.expecting_audio = True
            self.first_audio_chunk = True

            # Clear the recording for the next interaction
            self.recorder.clear_recording()

        except Exception as e:
            logger.exception(f"Error processing recorded audio: {e}")
            self.processing_input = False

    def run_conversation(self):
        """Run an interactive conversation with automatic voice detection"""
        try:
            print("\nStarting automatic voice detection with improved barge-in support.")
            print("You can interrupt the assistant by speaking while it's talking.")
            print("(Press Ctrl+C to exit)\n")

            # Start listening for voice
            self.recorder.start_listening()

            # Keep the main thread alive
            while self.connected:
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Exiting...")
        finally:
            self.disconnect()