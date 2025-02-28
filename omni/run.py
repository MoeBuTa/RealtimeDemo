import asyncio
from typing import Optional

from loguru import logger
import sounddevice as sd

from omni.realtime.audio import AudioStreamingApp


async def shutdown(
        app: AudioStreamingApp,
        input_stream: Optional[sd.InputStream]
) -> None:
    """
    Handle graceful shutdown.

    Args:
        app: The audio streaming application
        input_stream: Active sounddevice input stream, if any
    """
    logger.info("Shutting down application...")
    if app.loop:
        for task in asyncio.all_tasks(loop=app.loop):
            if task is not asyncio.current_task():
                task.cancel()

    # Close the agent connection
    await app.agent.close()

    # Close the input stream
    if input_stream is not None:
        input_stream.close()
        logger.debug("Closed input stream")

    # Stop any audio playback
    sd.stop()
    logger.debug("Stopped audio playback")
    logger.info("Shutdown complete")


async def async_run() -> None:
    """Application entry point with error handling and graceful shutdown."""
    logger.info("Starting realtime audio streaming application")

    app: AudioStreamingApp = AudioStreamingApp(model="gpt-4o-realtime-preview")
    input_stream: Optional[sd.InputStream] = None

    try:
        # Start the microphone input stream
        input_stream = sd.InputStream(
            callback=app.mic_callback,
            channels=app.channels,
            samplerate=app.sample_rate,
            blocksize=1024,  # Smaller blocks for lower latency
            dtype='float32'
        )

        with input_stream:
            logger.info("Microphone stream started")
            await app.run()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.exception(f"Application error: {e}")
    finally:
        await shutdown(app, input_stream)

def main():
    asyncio.run(async_run())