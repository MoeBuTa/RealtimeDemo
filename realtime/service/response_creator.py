# realtime/service/response_creator.py
from typing import List, Dict, Any, Optional, Callable
from loguru import logger


class ResponseCreator:
    """Service for creating response events"""

    @staticmethod
    def create(send_event_callback: Callable,
               default_instructions: str,
               state_updater: Optional[Callable] = None,
               **kwargs) -> None:
        """
        Create a response with customizable parameters

        Args:
            send_event_callback: Callback to send events to the WebSocket
            default_instructions: Default instructions if none provided
            state_updater: Callback to update client state
            **kwargs: Additional parameters including:
                - instructions: Custom instructions for the assistant
                - modalities: Response modalities (defaults to ["text", "audio"])
                - voice: Voice to use for audio responses
                - output_audio_format: Format for audio output
                - tools: List of tools available to the assistant
                - tool_choice: Tool selection strategy
                - temperature: Temperature setting for response generation
                - max_output_tokens: Maximum number of tokens to generate
        """
        # Extract parameters from kwargs or use defaults
        instructions = kwargs.get('instructions')
        modalities = kwargs.get('modalities', ["text", "audio"])
        voice = kwargs.get('voice', "sage")
        output_audio_format = kwargs.get('output_audio_format', "pcm16")
        tools = kwargs.get('tools')
        tool_choice = kwargs.get('tool_choice', "auto")
        temperature = kwargs.get('temperature', 0.8)
        max_output_tokens = kwargs.get('max_output_tokens', 1024)

        # Use provided instructions or fall back to default
        instructions_text = instructions if instructions is not None else default_instructions

        logger.info(f"Creating response with instructions: {instructions_text}")

        # Prepare response configuration
        response_config = {
            "modalities": modalities,
            "instructions": instructions_text,
            "voice": voice,
            "output_audio_format": output_audio_format,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens
        }

        # Add tools configuration if provided
        if tools:
            response_config["tools"] = tools
            response_config["tool_choice"] = tool_choice

        # Create a response with the configuration
        send_event_callback("response.create", {
            "response": response_config
        })

        # Update state to expect new audio
        if state_updater:
            state_updater({
                'expecting_audio': True,
                'first_audio_chunk': True
            })