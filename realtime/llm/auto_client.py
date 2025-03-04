import json
import threading
import time
from typing import Dict, Any

import websocket

from realtime.hardware.player import AudioPlayer
# Import our new automatic recorder instead of the original
from realtime.hardware.auto_recorder import AutomaticAudioRecorder
from realtime.utils.config import REALTIME_URL, HEADERS
from loguru import logger


class AutomaticRealtimeClient:
    """Client for the OpenAI Realtime API using direct WebSocket connection with automatic voice recording"""

    def __init__(self):
        self.ws = None
        self.player = AudioPlayer()
        # Use our automatic recorder instead
        self.recorder = AutomaticAudioRecorder(
            threshold=0.02,  # Adjust based on your microphone sensitivity
            silence_duration=1.0,  # Stop recording after 1 second of silence
            min_speech_duration=0.5,  # Require at least 0.5 seconds of speech
            pre_buffer_duration=0.5  # Keep 0.5 seconds before speech detection
        )
        self.connected = False
        self.current_event_id = 0
        self.processing_input = False  # Flag to prevent overlapping processing

    def connect(self):
        """Connect to the Realtime API"""

        def on_open(ws):
            self.connected = True
            logger.info("Connected to OpenAI Realtime API")

        def on_message(ws, message):
            try:
                event = json.loads(message)
                event_type = event.get("type")

                # Log the full event for debugging purposes
                try:
                    event_json = json.dumps(event, indent=2)
                    if len(event_json) > 1000:
                        logger.debug(f"Raw event (truncated): {event_json[:500]}...{event_json[-500:]}")
                    else:
                        logger.debug(f"Raw event: {event_json}")
                except:
                    logger.debug("Could not serialize event to JSON for logging")

                # Handle different event types
                if event_type == "audio.content":
                    logger.info("Received audio content event")
                    audio_content = event.get("audio_content", {})
                    audio_data = audio_content.get("audio", "")
                    logger.info(f"Processing audio content of size: {len(audio_data)} bytes")
                    self.player.add_audio(audio_data)

                elif event_type == "response.audio.delta":
                    logger.info("Received audio delta event")
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
                    # Reset processing flag when response is complete
                    self.processing_input = False

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

        # Set up recorder callback
        self.recorder.set_recording_finished_callback(self.process_recorded_audio)

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

            # Append the audio to the input buffer
            self.send_event("input_audio_buffer.append", {"audio": audio_data})

            # Commit the audio buffer
            self.send_event("input_audio_buffer.commit")

            # Create a response
            self.send_event("response.create")

            # Clear the recording for the next interaction
            self.recorder.clear_recording()

        except Exception as e:
            logger.exception(f"Error processing recorded audio: {e}")
            self.processing_input = False

    def run_conversation(self):
        """Run an interactive conversation with automatic voice detection"""
        try:
            print("\nStarting automatic voice detection. Just speak to interact...\n")
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