# realtime/handler/factory.py
from typing import Dict, Type, Optional
from loguru import logger

from realtime.handler.base import EventHandler
from realtime.handler.speech_started import SpeechStartedEventHandler
from realtime.handler.audio_buffer_committed import AudioBufferCommittedEventHandler
from realtime.handler.conversation_item_created import ConversationItemCreatedEventHandler
from realtime.handler.response_created import ResponseCreatedEventHandler
from realtime.handler.rate_limits_updated import RateLimitsUpdatedEventHandler
from realtime.handler.output_item_added import OutputItemAddedEventHandler
from realtime.handler.content_part_added import ContentPartAddedEventHandler
from realtime.handler.audio_delta import AudioDeltaEventHandler
from realtime.handler.audio_transcription_delta import AudioTranscriptDeltaEventHandler
from realtime.handler.error import ErrorEventHandler
from realtime.handler.response_done import ResponseDoneEventHandler
from realtime.handler.session_updated import SessionUpdatedEventHandler


class DefaultEventHandler(EventHandler):
    """Default handler for unknown event types"""

    def handle(self, event, **kwargs):
        """Handle unknown event types"""
        event_type = event.get("type")
        logger_fn = kwargs.get('logger', logger.info)
        logger_fn(f"Event: {event_type}")


class EventHandlerFactory:
    """Factory for creating and retrieving event handlers"""

    _handlers: Dict[str, EventHandler] = {}
    _default_handler: EventHandler = DefaultEventHandler()

    @classmethod
    def register_handler(cls, event_type: str, handler: EventHandler) -> None:
        """
        Register a handler for an event type

        Args:
            event_type: The type of event to handle
            handler: The handler instance
        """
        cls._handlers[event_type] = handler

    @classmethod
    def get_handler(cls, event_type: str) -> EventHandler:
        """
        Get the handler for an event type

        Args:
            event_type: The type of event

        Returns:
            The appropriate handler or the default handler
        """
        return cls._handlers.get(event_type, cls._default_handler)

    @classmethod
    def initialize_default_handlers(cls) -> None:
        """Initialize the default set of handlers"""
        # Register all the default handlers
        cls.register_handler("input_audio_buffer.speech_started", SpeechStartedEventHandler())
        cls.register_handler("input_audio_buffer.committed", AudioBufferCommittedEventHandler())
        cls.register_handler("conversation.item.created", ConversationItemCreatedEventHandler())
        cls.register_handler("response.created", ResponseCreatedEventHandler())
        cls.register_handler("rate_limits.updated", RateLimitsUpdatedEventHandler())
        cls.register_handler("response.output_item.added", OutputItemAddedEventHandler())
        cls.register_handler("response.content_part.added", ContentPartAddedEventHandler())
        cls.register_handler("response.audio.delta", AudioDeltaEventHandler())
        cls.register_handler("response.audio_transcript.delta", AudioTranscriptDeltaEventHandler())
        cls.register_handler("error", ErrorEventHandler())
        cls.register_handler("response.done", ResponseDoneEventHandler())
        cls.register_handler("session.updated", SessionUpdatedEventHandler())