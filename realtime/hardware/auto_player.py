import base64
import queue
import threading
import time as time_module
import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger
import io
import wave


class AutomaticAudioPlayer:
    """Handles streaming audio playback from the API responses with interruption support"""

    def __init__(self, buffer_size=10, sample_rate=24000):
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.playback_thread = None
        self.buffer_size = buffer_size  # Number of chunks to buffer before playback
        self.sample_rate = sample_rate  # Default sample rate for OpenAI audio
        self.audio_buffer = None  # NumPy buffer for continuous playback
        self.buffer_event = threading.Event()  # Signal when buffer is ready

        # Stream parameters
        self.stream = None
        self.stream_lock = threading.Lock()

    def start(self):
        """Start the audio playback thread"""
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.is_playing = True
            self.playback_thread = threading.Thread(target=self._playback_worker)
            self.playback_thread.daemon = True
            self.playback_thread.start()
            logger.info("Audio playback system ready")

    def stop(self):
        """Stop audio playback"""
        self.is_playing = False

        # Close the stream if it exists
        with self.stream_lock:
            if self.stream is not None and self.stream.active:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    logger.warning(f"Error stopping stream: {e}")
                finally:
                    self.stream = None

        if self.playback_thread and self.playback_thread.is_alive():
            try:
                self.playback_thread.join(1.0)
            except Exception as e:
                logger.warning(f"Error joining thread: {e}")

        logger.info("Audio playback stopped")

    def interrupt(self):
        """Interrupt current playback and clear queue for new conversation"""
        logger.info("Interrupting audio playback for new conversation")

        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Clear the buffer
        self.audio_buffer = np.array([])

        # Optional: Play a short interruption sound
        # self._play_interruption_sound()

        logger.info("Audio playback interrupted and queue cleared")

    def _play_interruption_sound(self):
        """Play a short sound to indicate interruption (optional)"""
        try:
            # Create a short beep sound
            duration = 0.2  # seconds
            frequency = 880  # Hz (A5)
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            beep = 0.2 * np.sin(2 * np.pi * frequency * t)  # Quiet beep

            # Fade in/out to avoid clicks
            fade = 0.05  # seconds
            fade_samples = int(fade * self.sample_rate)
            beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
            beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)

            # Play the beep directly
            sd.play(beep, self.sample_rate)
            sd.wait()
        except Exception as e:
            logger.warning(f"Could not play interruption sound: {e}")

    def add_audio(self, base64_audio: str):
        """Add base64 encoded audio to the playback queue for streaming"""
        try:
            audio_data = base64.b64decode(base64_audio)

            # Decode the audio data to numpy array for efficient processing
            samples = self._decode_audio(audio_data)
            if samples is not None:
                self.audio_queue.put(samples)
                logger.debug(f"Audio chunk added to playback queue: {len(samples)} samples")

        except Exception as e:
            logger.exception(f"Error decoding audio: {e}")

    def _decode_audio(self, audio_data):
        """Decode audio data to a numpy array"""
        try:
            # Check if data has WAV headers
            if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]:
                # Use soundfile to read WAV data directly from bytes
                with io.BytesIO(audio_data) as buf:
                    samples, _ = sf.read(buf)
                    return samples
            else:
                # Assume it's raw PCM 16-bit mono at 24kHz
                # Create a WAV in memory to parse it properly
                with io.BytesIO() as wav_io:
                    with wave.open(wav_io, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(self.sample_rate)
                        wav_file.writeframes(audio_data)

                    wav_io.seek(0)
                    samples, _ = sf.read(wav_io)
                    return samples

        except Exception as e:
            logger.exception(f"Error decoding audio data: {e}")
            return None

    def _callback(self, outdata, frames, time_info, status):
        """Callback for the sounddevice OutputStream

        Args:
            outdata: Output buffer as a numpy array
            frames: Number of frames to process
            time_info: Stream timing information (not used directly)
            status: Status flag indicating underflow/overflow
        """
        if status:
            logger.warning(f"Sounddevice status: {status}")

        # If we have data in our buffer, play it
        if self.audio_buffer is not None and len(self.audio_buffer) > 0:
            if len(self.audio_buffer) >= frames:
                # We have enough data
                outdata[:] = self.audio_buffer[:frames].reshape(-1, 1)
                # Remove the data we just played
                self.audio_buffer = self.audio_buffer[frames:]
            else:
                # Not enough data, pad with zeros
                outdata[:len(self.audio_buffer)] = self.audio_buffer.reshape(-1, 1)
                outdata[len(self.audio_buffer):] = 0
                self.audio_buffer = np.array([])

            if len(self.audio_buffer) == 0:
                # Signal that we need more data
                self.buffer_event.clear()
        else:
            # No data available, output silence
            outdata.fill(0)
            # Signal that we need data
            self.buffer_event.clear()

    def _playback_worker(self):
        """Worker thread that manages continuous audio playback"""
        try:
            # Initialize audio buffer
            self.audio_buffer = np.array([])

            # Create and start the output stream
            with self.stream_lock:
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=self._callback,
                    blocksize=1024  # Smaller blocksize for more responsive playback
                )
                self.stream.start()

            logger.info("Audio playback stream started")

            while self.is_playing:
                # Check if we need to fill the buffer
                if len(self.audio_buffer) < self.sample_rate * 0.5:  # Buffer less than 0.5 seconds
                    # Try to get more audio data
                    try:
                        # Non-blocking get with timeout
                        samples = self.audio_queue.get(timeout=0.1)

                        # Append to our buffer
                        if len(samples) > 0:
                            if len(self.audio_buffer) == 0:
                                self.audio_buffer = samples
                            else:
                                self.audio_buffer = np.append(self.audio_buffer, samples)

                            logger.debug(f"Buffer filled: {len(self.audio_buffer)} samples")
                            self.buffer_event.set()
                    except queue.Empty:
                        # No data available yet, wait a bit
                        time_module.sleep(0.01)
                else:
                    # Buffer is sufficiently full, wait for it to drain
                    self.buffer_event.wait(0.1)

        except Exception as e:
            logger.exception(f"Playback worker error: {e}")
        finally:
            # Ensure stream is closed on exit
            with self.stream_lock:
                if self.stream is not None:
                    try:
                        if self.stream.active:
                            self.stream.stop()
                        self.stream.close()
                    except Exception as e:
                        logger.warning(f"Error closing stream: {e}")
                    finally:
                        self.stream = None