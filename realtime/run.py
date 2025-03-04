from realtime.llm.auto_client import AutomaticRealtimeClient
from realtime.llm.client import RealtimeClient
from loguru import logger



# def main():
#     """Main function to run the Realtime client"""
#     logger.info("Starting OpenAI Realtime API client...")
#     client = RealtimeClient()
#
#     try:
#         client.connect()
#         client.run_conversation()
#     except Exception as e:
#         logger.info(f"Error: {e}")
#     finally:
#         client.disconnect()


def main():
    """Main function"""
    logger.info("Starting voice-activated realtime conversation")

    try:
        client = AutomaticRealtimeClient()
        client.connect()

        print("\n" + "=" * 60)
        print("Voice-Activated Realtime Conversation")
        print("=" * 60)
        print("\nJust start speaking to interact with the assistant.")
        print("The system will automatically detect when you start and stop talking.")
        print("\nPress Ctrl+C to exit.")
        print("=" * 60 + "\n")

        client.run_conversation()

    except Exception as e:
        logger.exception(f"Error in main: {e}")
    finally:
        print("\nExiting voice-activated conversation. Goodbye!")