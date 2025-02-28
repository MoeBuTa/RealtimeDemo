import asyncio
from asyncio import Queue, Task
import base64
from collections import deque
import time
from typing import Any, Coroutine, Deque, List, Optional, TypeVar, cast

# Import loguru for consistent logging
from loguru import logger
import numpy as np
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnectionManager
import sounddevice as sd

from omni.llm.realtime import RealtimeAgent

# Type aliases
AudioChunk = str  # Base64 encoded audio
NumpyAudio = np.ndarray  # Audio as numpy array
T = TypeVar('T')  # Generic type for asyncio gather


class AudioStreamingApp:
    def __init__(
            self,
            model: str = "gpt-4o-realtime-preview",
            agent_type: str = "realtime",
            sample_rate: int = 24000,
            channels: int = 1
    ) -> None:
        """
        Initialize the audio streaming application.

        Args:
            model: OpenAI model identifier
            agent_type: Type of agent to use
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono)
        """
        self.sample_rate: int = sample_rate
        self.channels: int = channels
        self.mic_audio_queue: Queue[AudioChunk] = asyncio.Queue()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Audio buffer for smoother playback
        self.audio_buffer: Deque[NumpyAudio] = deque(maxlen=10)  # Adjust buffer size as needed
        self.is_playing: bool = False

        # For audio quality monitoring
        self.last_audio_time: float = 0
        self.audio_chunks_received: int = 0

        # Initialize the agent
        self.agent = RealtimeAgent(model, agent_type)
        logger.info(f"AudioStreamingApp initialized with model {model}")

    def mic_callback(
            self,
            indata: np.ndarray,
            frames: int,
            time: Any,
            status: Optional[sd.CallbackFlags]
    ) -> None:
        """
        Callback for processing microphone input.

        Args:
            indata: Audio data from microphone
            frames: Number of frames
            time: Time info from sounddevice
            status: Status flags
        """
        if status:
            logger.warning(f"Microphone status: {status}")

        # Convert float32 audio (range [-1,1]) to int16
        audio_int16: np.ndarray = np.int16(indata * 32767)
        audio_bytes: bytes = audio_int16.tobytes()
        audio_b64: str = base64.b64encode(audio_bytes).decode('utf-8')

        # Schedule adding the audio chunk to the asyncio queue
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.mic_audio_queue.put(audio_b64), self.loop)

    async def send_mic_audio(self, connection: Any) -> None:
        """
        Send microphone audio to the API using a simpler polling approach that works better on macOS.

        Args:
            connection: OpenAI realtime connection (not the manager)
        """
        batch_size: int = 3  # Number of chunks to batch together
        batch: List[AudioChunk] = []
        audio_device = None  # Use default device

        try:
            # List devices to help troubleshoot
            devices = sd.query_devices()
            logger.info(f"Available audio devices: {len(devices)}")
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    logger.info(f"Input device {i}: {dev['name']} (in={dev['max_input_channels']})")
                    if audio_device is None:
                        # Use the first input device found
                        audio_device = i
                        logger.info(f"Selected device {i} as default input")

            # Try a different approach without callbacks
            logger.info(f"Starting audio recording with device {audio_device}, rate {self.sample_rate}Hz")

            # Start a loop to record and send audio
            while True:
                try:
                    # Record a chunk of audio (200ms)
                    duration = 0.2
                    logger.info("Recording audio chunk...")

                    # Use simpler blocking recording function instead of callback
                    recorded_audio = sd.rec(
                        int(duration * self.sample_rate),
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        device=audio_device,
                        dtype='float32'
                    )

                    # Wait for the recording to complete
                    sd.wait()

                    # Convert to base64
                    audio_int16 = np.int16(recorded_audio * 32767)
                    audio_bytes = audio_int16.tobytes()
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                    # Add to batch
                    batch.append(audio_b64)

                    # Send when batch is full
                    if len(batch) >= batch_size:
                        combined_audio = "".join(batch)
                        await self.agent.send_audio(combined_audio)
                        logger.info(f"Sent audio to OpenAI ({len(combined_audio)} bytes)")
                        batch = []

                    # Small pause to prevent overwhelming
                    await asyncio.sleep(0.05)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error during audio recording: {e}")
                    await asyncio.sleep(1)  # Wait a bit before retrying

        except asyncio.CancelledError:
            logger.info("Audio sending task cancelled")
        except Exception as e:
            logger.error(f"Error in send_mic_audio: {e}")
        finally:
            # Send any remaining audio
            if batch:
                try:
                    combined_audio = "".join(batch)
                    await self.agent.send_audio(combined_audio)
                    logger.info(f"Sent final audio batch ({len(combined_audio)} bytes)")
                except Exception as e:
                    logger.error(f"Failed to send final batch: {e}")

            # Text fallback if audio fails
            try:
                await self.agent.send_text("Audio may be unavailable. Please continue with text.")
            except Exception:
                pass

    async def play_buffered_audio(self) -> None:
        """Play audio from buffer for smoother playback."""
        try:
            while True:
                if self.audio_buffer and not self.is_playing:
                    self.is_playing = True
                    audio_data: NumpyAudio = self.audio_buffer.popleft()

                    # Play the chunk
                    sd.play(audio_data, samplerate=self.sample_rate)
                    # Wait for playback to finish
                    sd.wait()
                    self.is_playing = False
                else:
                    # Small sleep to prevent CPU hogging
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            sd.stop()
        except Exception as e:
            logger.error(f"Error in play_buffered_audio: {e}")
            sd.stop()

    async def process_response_audio(self, connection: Any) -> None:
        """
        Process audio responses from the API.

        Args:
            connection: OpenAI realtime connection (not the manager)
        """
        try:
            logger.info("Listening for OpenAI responses")
            # Use the actual connection object for event iteration
            async for event in connection:
                event_type: str = cast(str, event.type)

                if event_type == 'response.audio.delta':
                    self.audio_chunks_received += 1
                    current_time: float = time.time()

                    # Monitor audio stream health
                    if self.last_audio_time > 0:
                        time_diff: float = current_time - self.last_audio_time
                        if time_diff > 0.5:  # Potential audio gap
                            logger.warning(f"Audio gap detected: {time_diff:.2f}s")

                    self.last_audio_time = current_time

                    # Decode and process audio
                    audio_b64: str = cast(str, event.delta)
                    audio_bytes: bytes = base64.b64decode(audio_b64)
                    audio_int16: np.ndarray = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float: np.ndarray = audio_int16.astype(np.float32) / 32767.0

                    # Add to buffer for smoother playback
                    self.audio_buffer.append(audio_float)

                elif event_type == 'response.text.delta':
                    # Handle text deltas if needed
                    text_delta = cast(str, event.delta)
                    logger.info(f"Text from OpenAI: {text_delta}")

                elif event_type == 'response.audio.done':
                    logger.info(f"Received complete audio response ({self.audio_chunks_received} chunks)")
                    self.audio_chunks_received = 0

                elif event_type == "response.done":
                    logger.info("OpenAI response complete")
                    # Don't break here - keep listening for more events
                    # Just restart the response
                    await self.agent.start_response()
                    logger.info("Started new response cycle")

                elif event_type == "error":
                    logger.error(f"API error received: {event}")
                    # Try to restart after error
                    await asyncio.sleep(1)
                    await self.agent.start_response()
                    logger.info("Restarted response after error")

        except asyncio.CancelledError:
            logger.info("Response processing task cancelled")
        except Exception as e:
            logger.error(f"Error in process_response_audio: {e}")

    async def run(self) -> None:
        """Main application entry point."""
        self.loop = asyncio.get_running_loop()
        logger.info(f"Starting audio streaming with model: {self.agent.name}")

        # Initialize task variables to None
        send_task: Optional[Task[None]] = None
        process_task: Optional[Task[None]] = None
        play_task: Optional[Task[None]] = None

        try:
            # Connect to the API using the agent
            connection = await self.agent.connect()
            logger.info("Connected to OpenAI Realtime API")

            # Start playback task first
            play_task = asyncio.create_task(self.play_buffered_audio())

            # Start the response processing task
            process_task = asyncio.create_task(self.process_response_audio(connection))

            # Send initial message to start conversation
            await self.agent.send_text("Starting audio conversation")
            await self.agent.start_response()
            logger.info("Sent initial message to OpenAI")

            # Start the audio sending task last
            send_task = asyncio.create_task(self.send_mic_audio(connection))

            # Wait for tasks to complete or be cancelled
            await asyncio.gather(send_task, process_task, play_task)

        except asyncio.CancelledError:
            logger.info("Application shutdown initiated")
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
        finally:
            # Cancel any tasks that are still running
            for task in [t for t in [send_task, process_task, play_task] if t is not None]:
                if not task.done():
                    task.cancel()

            # Wait for tasks to properly cancel
            pending_tasks = [t for t in [send_task, process_task, play_task]
                             if t is not None and not t.done()]

            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)

            # Close the agent connection
            await self.agent.close()
            logger.info("Application shutdown complete")