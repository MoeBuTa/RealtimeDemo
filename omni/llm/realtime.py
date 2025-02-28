from typing import Any, Optional, cast
import json

from loguru import logger
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnectionManager

from omni.llm.base import BaseAgent
from omni.utils.config import OPENAI_API_KEY


class RealtimeAgent(BaseAgent):
    def __init__(self, name: str, agent_type: str, api_key: Optional[str] = None):
        """
        Initialize CloudAgent with model name and optional API key.

        Args:
            name: Name of the model to use (e.g., 'gpt-4o-realtime-preview')
            agent_type: Type of agent
            api_key: Optional OpenAI API key. If not provided, will look for
                    OPENAI_API_KEY in environment variables or .env file
        """
        super().__init__(name, agent_type)
        self.client = self._init_openai_client()
        self.connection_manager: Optional[AsyncRealtimeConnectionManager] = None
        self.connection: Optional[Any] = None

    def _init_openai_client(self) -> AsyncOpenAI:
        """
        Initialize OpenAI client with API key from either:
        1. Explicitly passed api_key parameter
        2. Environment variable OPENAI_API_KEY
        3. .env file
        """
        try:
            # Initialize client with config
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            logger.info(f"Successfully initialized OpenAI client for model {self.name}")
            return client

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    async def connect(self) -> AsyncRealtimeConnectionManager:
        """
        Establish connection to the OpenAI realtime API.

        Returns:
            Realtime connection object
        """
        if not self.connection:
            # Create the connection manager
            self.connection_manager = self.client.beta.realtime.connect(model=self.name)
            logger.info(f"Created connection manager for model {self.name}")

            # Use enter() method to get the actual connection
            self.connection = await self.connection_manager.enter()
            logger.info("Entered connection manager successfully")

            # Enable both audio and text modalities
            try:
                await self.connection.session.update(session={'modalities': ['audio', 'text']})
                logger.info(f"Connected to OpenAI API with model {self.name} and audio+text modalities enabled")
            except Exception as e:
                logger.error(f"Failed to update session modalities: {str(e)}")
                raise

        return self.connection

    async def send_audio(self, audio_data: str) -> None:
        """
        Send audio data to the realtime API.

        Args:
            audio_data: Base64 encoded audio data
        """
        if not self.connection:
            raise RuntimeError("Connection not established. Call connect() first.")

        try:
            logger.info(f"Sending audio data ({len(audio_data)} bytes)")
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "audio": audio_data}]
                }
            )
            logger.info("Audio data sent successfully")
        except Exception as e:
            logger.error(f"Failed to send audio data: {str(e)}")
            raise

    async def send_text(self, text: str) -> None:
        """
        Send text data to the realtime API.

        Args:
            text: Text message to send
        """
        if not self.connection:
            raise RuntimeError("Connection not established. Call connect() first.")

        try:
            logger.info(f"Sending text message: '{text}'")
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}]
                }
            )
            logger.info("Text message sent successfully")
        except Exception as e:
            logger.error(f"Failed to send text message: {str(e)}")
            raise

    async def start_response(self) -> None:
        """Trigger the model to generate a response."""
        if not self.connection:
            raise RuntimeError("Connection not established. Call connect() first.")

        try:
            logger.info("Requesting model response")
            await self.connection.response.create()
            logger.info("Response request sent successfully")
        except Exception as e:
            logger.error(f"Failed to start response: {str(e)}")
            raise

    async def process(self, message: str, **kwargs: Any) -> Any:
        """
        Process a message using the realtime API.

        Args:
            message: Text message to process
            **kwargs: Additional keyword arguments

        Returns:
            Response from the API
        """
        # Create a connection manager
        logger.info(f"Processing message with model {self.name}")
        connection_manager = self.client.beta.realtime.connect(model=self.name)

        # Use async with to properly manage the connection lifecycle
        async with await connection_manager.__aenter__() as connection:
            # Update the session settings
            logger.info("Setting up session with text and audio modalities")
            await connection.session.update(session={'modalities': ['text', 'audio']})

            # Create a conversation item (a message) with the text prompt
            logger.info(f"Sending text message: '{message}'")
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": message}
                    ],
                }
            )

            # Trigger the model to generate a response
            logger.info("Requesting model response")
            await connection.response.create()

            full_response = ""
            # Stream and process events from the realtime API
            event_count = 0
            logger.info("Waiting for events from OpenAI...")

            async for event in connection:
                event_count += 1
                # Log the full event object for inspection
                try:
                    # Try to convert the event to a dictionary for better logging
                    event_dict = {
                        "type": event.type,
                        "id": getattr(event, "id", None),
                        "delta": getattr(event, "delta", None),
                        "timestamp": getattr(event, "timestamp", None),
                    }
                    # Filter out None values
                    event_dict = {k: v for k, v in event_dict.items() if v is not None}
                    logger.info(f"Event {event_count}: {json.dumps(event_dict)}")
                except Exception as e:
                    # Fallback to just logging the event type if conversion fails
                    logger.info(f"Event {event_count}: {event.type} (could not serialize full event)")

                if event.type == 'response.text.delta':
                    # Accumulate text chunks
                    delta = cast(str, event.delta)
                    full_response += delta
                    logger.info(f"Text delta: '{delta}'")
                    print(delta, end="", flush=True)

                elif event.type == 'response.audio.delta':
                    # Log audio chunk info
                    audio_delta = cast(str, event.delta)
                    logger.info(f"Audio delta received: {len(audio_delta)} bytes")

                elif event.type == 'response.text.done':
                    # Log when text response is complete
                    logger.info(f"Text response complete: '{full_response}'")
                    print("\nResponse complete.")

                elif event.type == 'response.audio.done':
                    # Log when audio response is complete
                    logger.info("Audio response complete")

                elif event.type == "response.done":
                    # Log when full response is done
                    logger.info("Full response complete")
                    break

                elif event.type == "error":
                    # Log errors in detail
                    logger.error(f"Error event received: {event}")

            logger.info(f"Received {event_count} events in total")
            return full_response

    async def close(self) -> None:
        """Close the connection if open."""
        if self.connection:
            logger.info("Closing connection")
            await self.connection.close()
            self.connection = None
            self.connection_manager = None
            logger.info("Connection closed successfully")