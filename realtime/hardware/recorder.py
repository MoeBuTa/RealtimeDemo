import base64

import threading
import time

from loguru import logger
import numpy as np
import sounddevice as sd

from realtime.utils.config import MAX_RECORD_SECONDS, CHANNELS, RATE


class AudioRecorder:
    """Records audio from the Mac's built-in microphone"""

    def __init__(self):
        self.is_recording = False
        self.frames = []
        self.recording_thread = None

    def start(self):
        """Start recording audio"""
        self.is_recording = True
        self.frames = []
        self.recording_thread = threading.Thread(target=self._record_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        logger.info("Recording started...")

    def stop(self):
        """Stop recording audio"""
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(1.0)  # Wait for thread to finish
        logger.info("Recording stopped.")

    def _record_worker(self):
        """Worker function for audio recording"""

        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Recording status: {status}")
            self.frames.append(indata.copy())

        try:
            with sd.InputStream(callback=callback, channels=CHANNELS, samplerate=RATE):
                start_time = time.time()
                while self.is_recording:
                    # Safety check to avoid recording indefinitely
                    if time.time() - start_time > MAX_RECORD_SECONDS:
                        logger.warning(f"Maximum recording time of {MAX_RECORD_SECONDS} seconds reached.")
                        self.is_recording = False
                        break
                    sd.sleep(100)  # Sleep to reduce CPU usage
        except Exception as e:
            logger.exception(f"Error recording: {e}")

    def get_audio_data(self) -> bytes:
        """Get all recorded audio data as PCM bytes"""
        if not self.frames:
            return b''

        # Convert numpy arrays to a single array
        audio_data = np.concatenate(self.frames, axis=0)

        # Convert to int16 format (required by OpenAI)
        audio_data = (audio_data * 32767).astype(np.int16)

        # Return as bytes
        return audio_data.tobytes()

    def get_base64_audio(self) -> str:
        """Get the recorded audio as base64 encoded string"""
        return base64.b64encode(self.get_audio_data()).decode('utf-8')