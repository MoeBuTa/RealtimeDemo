import base64
import threading
import numpy as np
import sounddevice as sd
import time as time_module
import collections
import io
import wave
from loguru import logger


class AutomaticAudioRecorder:
    """Records audio automatically when speech is detected with support for barge-in detection"""

    def __init__(self, threshold=0.02, sample_rate=16000, channels=1,
                 silence_duration=1.0, min_speech_duration=0.5,
                 pre_buffer_duration=0.5):
        """
        Initialize the automatic audio recorder

        Args:
            threshold: Voice activity detection threshold
            sample_rate: Audio sample rate
            channels: Number of audio channels
            silence_duration: Duration of silence in seconds to end recording
            min_speech_duration: Minimum speech duration in seconds to consider valid
            pre_buffer_duration: Duration of audio to keep before speech is detected
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.channels = channels

        # Convert time-based parameters to frame counts
        # Assuming typical frame size of 1024 samples
        self.frame_size = 1024
        frames_per_second = sample_rate / self.frame_size

        self.silence_threshold = int(silence_duration * frames_per_second)
        self.min_speech_frames = int(min_speech_duration * frames_per_second)
        self.pre_buffer_size = int(pre_buffer_duration * frames_per_second)

        logger.info(f"Speech detection parameters: silence_threshold={self.silence_threshold} frames, "
                    f"min_speech_frames={self.min_speech_frames} frames, "
                    f"pre_buffer_size={self.pre_buffer_size} frames")

        # Initialize state variables
        self.is_recording = False
        self.is_listening = False
        self.recorded_frames = []
        self.listen_thread = None
        self.stream = None
        self.speech_detected = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.speech_start_time = None

        # Pre-buffer to keep audio before speech is detected
        self.pre_buffer = collections.deque(maxlen=self.pre_buffer_size)

        # Callbacks for various events
        self.recording_finished_callback = None
        self.speech_detected_callback = None

    def set_recording_finished_callback(self, callback):
        """Set callback function to be called when recording is finished"""
        self.recording_finished_callback = callback

    def set_speech_detected_callback(self, callback):
        """Set callback function to be called when speech is first detected (for barge-in)"""
        self.speech_detected_callback = callback

    def start_listening(self):
        """Start listening for speech"""
        if self.is_listening:
            return

        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_worker)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        logger.info("Started listening for speech")

    def stop_listening(self):
        """Stop listening for speech"""
        self.is_listening = False

        # Safely handle the stream
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.warning(f"Error stopping stream: {e}")
            finally:
                self.stream = None

        # Safely join the thread
        if hasattr(self, 'listen_thread') and self.listen_thread is not None and self.listen_thread.is_alive():
            try:
                self.listen_thread.join(timeout=1.0)
            except Exception as e:
                logger.warning(f"Error joining thread: {e}")

        logger.info("Stopped listening for speech")

    def is_speech_active(self):
        """Return whether speech is currently detected"""
        return self.speech_detected

    def _listen_worker(self):
        """Worker thread that listens for speech"""
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.frame_size,  # Ensure consistent block size
                callback=self.audio_callback
            )

            self.stream.start()

            # Keep thread alive as long as we're listening
            while self.is_listening:
                time_module.sleep(0.1)

        except Exception as e:
            logger.exception(f"Error in listen worker: {e}")
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        """Process incoming audio to detect speech"""
        if status:
            logger.warning(f"Input stream status: {status}")

        # Calculate volume (RMS)
        volume_norm = np.linalg.norm(indata) / np.sqrt(frames)

        # Always add current frame to pre-buffer
        self.pre_buffer.append(indata.copy())

        # Speech detection logic
        if volume_norm > self.threshold:
            if not self.speech_detected:
                logger.info(f"Speech detected (volume: {volume_norm:.4f})")
                self.speech_detected = True
                self.speech_start_time = time_module.time()
                self.speech_frames = 1

                # Call the speech detected callback for barge-in detection
                if self.speech_detected_callback:
                    # Run in separate thread to avoid blocking audio callback
                    threading.Thread(target=self.speech_detected_callback).start()

                # Start recording if not already
                if not self.is_recording:
                    self.start_recording()

                    # Add pre-buffer to recording
                    for frame in self.pre_buffer:
                        self.recorded_frames.append(frame)
            else:
                # Count consecutive speech frames
                self.speech_frames += 1

            # Reset silence counter
            self.silence_frames = 0
        else:
            # If we were in speech mode, count silent frames
            if self.speech_detected:
                self.silence_frames += 1

                # If we've had enough silence, end speech
                if self.silence_frames > self.silence_threshold:
                    duration = time_module.time() - self.speech_start_time
                    logger.info(f"Speech ended (duration: {duration:.2f}s, frames: {self.speech_frames})")

                    # Only consider it valid if it meets minimum duration
                    if self.speech_frames >= self.min_speech_frames:
                        logger.info(f"Valid speech detected ({self.speech_frames} frames)")

                        # Stop recording and process audio
                        if self.is_recording:
                            self.stop_recording()

                            # Call the callback if set
                            if self.recording_finished_callback:
                                threading.Thread(target=self.recording_finished_callback).start()
                    else:
                        logger.info(f"Speech too short ({self.speech_frames} frames), ignoring")
                        self.clear_recording()

                    # Reset speech detection state
                    self.speech_detected = False
                    self.speech_frames = 0

        # If we're recording, save the frames
        if self.is_recording:
            self.recorded_frames.append(indata.copy())

    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return

        self.is_recording = True
        self.recorded_frames = []
        logger.info("Started recording")

    def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            return

        self.is_recording = False
        logger.info(f"Stopped recording, captured {len(self.recorded_frames)} frames")

    def clear_recording(self):
        """Clear the current recording buffer"""
        self.recorded_frames = []
        self.is_recording = False
        logger.info("Recording cleared")

    def get_base64_audio(self):
        """Get the recorded audio as base64 encoded string"""
        if not self.recorded_frames or len(self.recorded_frames) == 0:
            logger.warning("No audio recorded")
            return None

        # Concatenate all frames
        audio_data = np.concatenate(self.recorded_frames, axis=0)

        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)

        # Create a WAV file in memory
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())

            # Get the WAV data and encode to base64
            wav_data = wav_io.getvalue()
            base64_audio = base64.b64encode(wav_data).decode('utf-8')

        logger.info(f"Encoded {len(wav_data)} bytes of audio to base64")
        return base64_audio