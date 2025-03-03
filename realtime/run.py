from realtime.llm.client import RealtimeClient
from loguru import logger



def main():
    """Main function to run the Realtime client"""
    logger.info("Starting OpenAI Realtime API client...")
    client = RealtimeClient()

    try:
        client.connect()
        client.run_conversation()
    except Exception as e:
        logger.info(f"Error: {e}")
    finally:
        client.disconnect()