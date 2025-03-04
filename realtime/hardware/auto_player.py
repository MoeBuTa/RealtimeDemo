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
    """Handles streaming audio playback from the API responses with improved force restart capability"""

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

        # Flag to silence output but keep everything running
        self.silence_output = False

        # Added thread management
        self.thread_id = 0  # To track thread instances
        self.current_thread_id = None  # Current active thread

    def force_restart(self):
        """Forcefully stop current playback and restart the player completely"""
        logger.info("Forcefully restarting audio player")

        # Force stop the current player completely
        with self.stream_lock:
            # Mark current thread as stopped
            self.is_playing = False

            # Close the stream if it exists
            if self.stream is not None and self.stream.active:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    logger.warning(f"Error stopping stream: {e}")
                finally:
                    self.stream = None

            # Clear the queue completely
            self._clear_queue()

            # Reset audio buffer
            self.audio_buffer = np.array([])

            # Create a new thread ID
            self.thread_id += 1
            new_thread_id = self.thread_id

            # Start a new playback thread with the new ID
            self.is_playing = True
            self.silence_output = False
            self.current_thread_id = new_thread_id
            self.playback_thread = threading.Thread(
                target=self._playback_worker,
                args=(new_thread_id,),
                name=f"playback-{new_thread_id}"
            )
            self.playback_thread.daemon = True
            self.playback_thread.start()

            logger.info(f"Started new playback thread ID: {new_thread_id}")

    def start(self):
        """Start the audio playback thread"""
        with self.stream_lock:
            if self.playback_thread is None or not self.playback_thread.is_alive():
                self.is_playing = True
                self.silence_output = False
                self.thread_id += 1
                self.current_thread_id = self.thread_id
                self.playback_thread = threading.Thread(
                    target=self._playback_worker,
                    args=(self.current_thread_id,),
                    name=f"playback-{self.current_thread_id}"
                )
                self.playback_thread.daemon = True
                self.playback_thread.start()
                logger.info(f"Audio playback system ready (thread ID: {self.current_thread_id})")

    def stop(self):
        """Stop audio playback completely (for shutdown only)"""
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

    def _clear_queue(self):
        """Clear the audio queue"""
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
        except Exception:
            pass

    def stop_current_audio(self):
        """Simple approach to stop current audio without stopping the process"""
        logger.info("Stopping current audio playback")

        # Set flag to silence output
        self.silence_output = True

        # Clear the queue
        self._clear_queue()

        # Clear the buffer
        with self.stream_lock:
            self.audio_buffer = np.array([])

        # Reset flag after a short delay
        def reset_flag():
            time_module.sleep(0.2)  # Short delay to ensure audio is cleared
            self.silence_output = False
            logger.info("Audio system ready for new audio")

        # Start thread to reset flag
        threading.Thread(target=reset_flag, daemon=True).start()

    def add_audio(self, base64_audio: str):
        """Add base64 encoded audio to the playback queue for streaming"""
        # Skip if we're silencing output
        if self.silence_output:
            return

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
        """Callback for the sounddevice OutputStream"""
        if status:
            logger.warning(f"Sounddevice status: {status}")

        # If silencing, output zeros
        if self.silence_output:
            outdata.fill(0)
            return

        # Use lock when accessing the buffer
        with self.stream_lock:
            # If we have data in our buffer, play it
            if self.audio_buffer is not None and len(self.audio_buffer) > 0 and not self.silence_output:
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

    def _playback_worker(self, thread_id):
        """Worker thread that manages continuous audio playback"""
        try:
            logger.info(f"Playback worker started with thread ID: {thread_id}")

            # Check if this is still the current thread
            if thread_id != self.current_thread_id:
                logger.info(f"Thread {thread_id} is obsolete, exiting")
                return

            # Initialize audio buffer
            self.audio_buffer = np.array([])

            # Create and start the output stream
            with self.stream_lock:
                # Check again if this thread is still current
                if thread_id != self.current_thread_id:
                    logger.info(f"Thread {thread_id} is obsolete before stream creation, exiting")
                    return

                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=self._callback,
                    blocksize=1024  # Smaller blocksize for more responsive playback
                )
                self.stream.start()

            logger.info(f"Audio playback stream started for thread {thread_id}")

            while self.is_playing and thread_id == self.current_thread_id:
                # Check if our thread is still current
                if thread_id != self.current_thread_id:
                    logger.info(f"Thread {thread_id} is no longer current, exiting")
                    break

                # Don't get new audio while silencing
                if self.silence_output:
                    time_module.sleep(0.05)
                    continue

                # Check if we need to fill the buffer
                buffer_size = 0
                with self.stream_lock:
                    buffer_size = len(self.audio_buffer)

                if buffer_size < self.sample_rate * 0.5:  # Buffer less than 0.5 seconds
                    # Try to get more audio data
                    try:
                        # Non-blocking get with timeout
                        samples = self.audio_queue.get(timeout=0.1)

                        # Skip if we started silencing or thread is obsolete
                        if self.silence_output or thread_id != self.current_thread_id:
                            continue

                        # Append to our buffer
                        if len(samples) > 0:
                            with self.stream_lock:
                                if len(self.audio_buffer) == 0:
                                    self.audio_buffer = samples
                                else:
                                    self.audio_buffer = np.append(self.audio_buffer, samples)

                            logger.debug(f"Buffer filled: {len(samples)} samples")
                            self.buffer_event.set()
                    except queue.Empty:
                        # No data available yet, wait a bit
                        time_module.sleep(0.01)
                else:
                    # Buffer is sufficiently full, wait for it to drain
                    self.buffer_event.wait(0.1)

        except Exception as e:
            logger.exception(f"Playback worker error in thread {thread_id}: {e}")
        finally:
            # Ensure stream is closed on exit if this is still the current thread
            with self.stream_lock:
                if thread_id == self.current_thread_id and self.stream is not None and self.stream.active:
                    try:
                        self.stream.stop()
                        self.stream.close()
                        logger.info(f"Stream closed by thread {thread_id}")
                    except Exception as e:
                        logger.warning(f"Error stopping stream in thread {thread_id}: {e}")
                    finally:
                        self.stream = None

            logger.info(f"Playback worker thread {thread_id} exited")