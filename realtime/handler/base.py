# realtime/handler/base.py
import abc
from typing import Any, Dict


class EventHandler(abc.ABC):
    """Base abstract class for event handlers"""

    @abc.abstractmethod
    def handle(self, event: Dict[str, Any], **kwargs) -> None:
        """
        Handle an event from the WebSocket

        Args:
            event: The event data from the websocket
            **kwargs: Additional context provided by the client
        """
        pass