from loguru import logger

from realtime.run import main

if __name__ == '__main__':
    try:
        # Configure loguru (optional - you can set this up based on your preferences)
        logger.remove()  # Remove default handler
        logger.add(
            "audio_streaming.log",
            rotation="10 MB",
            retention="1 week",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )
        # Also log to console
        logger.add(
            lambda msg: print(msg, end=""),
            level="INFO",
            format="{time:HH:mm:ss} | <level>{level: <8}</level> | {message}",
            colorize=True,
        )

        logger.info("Starting application")
        main()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
