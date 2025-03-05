# realtime/service/bargein_handler.py
import time
from typing import Dict, Any, Callable

from loguru import logger


class BargeinHandler:
    """Service for handling barge-in events"""

    @staticmethod
    def handle(send_event_callback: Callable,
               player: Any,
               state_updater: Callable,
               client_state: Dict[str, Any]) -> None:
        """
        Handle barge-in with complete player stopping

        Args:
            send_event_callback: Function to send events
            player: Audio player instance
            state_updater: Function to update client state
            client_state: Current client state
        """
        if client_state.get('assistant_speaking', False):
            logger.info("BARGE-IN DETECTED: User started speaking while assistant was talking")

            # Cancel the current response if there is one
            current_response_id = client_state.get('current_response_id')
            if current_response_id:
                logger.info(f"Canceling response {current_response_id} due to barge-in")
                try:
                    send_event_callback("response.cancel", {"response_id": current_response_id})
                except Exception as e:
                    logger.warning(f"Error canceling response: {e}")

            # Force kill the player completely
            player.stop()

            # Wait a small amount of time to ensure clean shutdown
            time.sleep(0.1)

            # Start a new player instance
            player.start()

            # Reset assistant speaking flag
            state_updater({'assistant_speaking': False})

            # Show visual indicator of barge-in
            print("\n[User interrupted - listening...]")