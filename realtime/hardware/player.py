import base64

import os
import queue
import threading
import time

import sounddevice as sd
import soundfile as sf
from loguru import logger


class AudioPlayer:
    """Handles streaming audio playback from the API responses"""

    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.playback_thread = None

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
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(1.0)
        logger.info("Audio playback stopped")

    def add_audio(self, base64_audio: str):
        """Add base64 encoded audio to the playback queue for streaming"""
        try:
            # Log audio content size for debugging
            logger.info(f"Received audio chunk of size: {len(base64_audio)} bytes")

            audio_data = base64.b64decode(base64_audio)
            logger.info(f"Decoded audio data size: {len(audio_data)} bytes")

            self.audio_queue.put(audio_data)
            logger.info("Audio chunk added to playback queue")
        except Exception as e:
            logger.exception(f"Error decoding audio: {e}")

    def _playback_worker(self):
        """Worker thread that streams audio from the queue to the speakers"""

        # Function to save audio data to WAV file with proper headers
        def save_audio_to_wav(audio_data, filename):
            try:
                import wave
                import struct

                # Define WAV parameters
                channels = 1  # Mono
                sample_width = 2  # 2 bytes per sample (16-bit)
                sample_rate = 24000  # OpenAI default is 24kHz

                with wave.open(filename, 'wb') as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(sample_width)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)

                logger.info(f"Audio saved to {filename} with proper WAV headers")
                return True
            except Exception as e:
                logger.exception(f"Error saving audio to WAV: {e}")
                return False

        try:
            while self.is_playing:
                if not self.audio_queue.empty():
                    logger.info("Getting audio chunk from queue for playback")
                    audio_data = self.audio_queue.get()
                    logger.info(f"Processing audio chunk of size: {len(audio_data)} bytes")

                    # Determine if this is raw audio data or already has WAV headers
                    has_wav_header = audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]
                    logger.info(f"Audio data {'has' if has_wav_header else 'does not have'} WAV headers")

                    # Create a temporary file to write the audio data
                    temp_file = "temp_audio.wav"

                    if has_wav_header:
                        # Direct write if it already has WAV headers
                        with open(temp_file, "wb") as f:
                            f.write(audio_data)
                        logger.info(f"Audio data with WAV headers written to file: {temp_file}")
                    else:
                        # Save with proper WAV headers if it's raw audio
                        success = save_audio_to_wav(audio_data, temp_file)
                        if not success:
                            logger.warning("Falling back to direct write method")
                            with open(temp_file, "wb") as f:
                                f.write(audio_data)

                    # For debugging, save a copy of the audio file
                    debug_file = f"debug_audio_{int(time.time())}.wav"
                    import shutil
                    shutil.copy(temp_file, debug_file)
                    logger.info(f"Debug copy saved to {debug_file}")

                    # Read and play the audio
                    try:
                        logger.info("Reading audio file with soundfile...")
                        data, samplerate = sf.read(temp_file)
                        logger.info(f"Audio file read: {data.shape} samples at {samplerate}Hz")

                        logger.info("Playing audio through sounddevice...")
                        sd.play(data, samplerate)
                        sd.wait()  # Wait until audio is done playing
                        logger.info("Audio playback completed")
                    except Exception as e:
                        logger.exception(f"Error playing audio: {e}")

                        # Try alternative playback method if the first one fails
                        try:
                            logger.info("Trying alternative playback method with afplay (Mac OS X)...")
                            import subprocess
                            subprocess.run(["afplay", temp_file], check=True)
                            logger.info("Alternative playback completed")
                        except Exception as alt_e:
                            logger.exception(f"Alternative playback also failed: {alt_e}")

                    # Clean up the temporary file
                    try:
                        os.remove(temp_file)
                        logger.debug(f"Temporary file {temp_file} removed")
                    except Exception as e:
                        logger.warning(f"Could not remove temp file: {e}")
                else:
                    time.sleep(0.05)  # Short sleep to reduce CPU usage
        except Exception as e:
            logger.exception(f"Playback error: {e}")
