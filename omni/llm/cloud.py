import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Type, Union

from openai import OpenAI
from openai._types import NotGiven
from openai.lib import ResponseFormatT
from openai.types.chat import ChatCompletionMessageParam, completion_create_params

from omni.llm.base import BaseAgent
from omni.llm.config import CLOUD_MODEL_CONFIGS
from omni.utils.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)


class CloudAgent(BaseAgent):
    def __init__(self, name: str, agent_type: str, api_key: Optional[str] = None):
        """
        Initialize CloudAgent with model name and optional API key.

        Args:
            name: Name of the model to use (e.g., 'gpt-4')
            api_key: Optional OpenAI API key. If not provided, will look for
                    OPENAI_API_KEY in environment variables or .env file
        """
        super().__init__(name, agent_type)
        self.model = self._init_model_config()
        self.client = self._init_openai_client()

    def _init_model_config(self) -> Dict[str, Any]:
        """Initialize model configuration from predefined configs"""
        model_name = self.name.lower()
        config = CLOUD_MODEL_CONFIGS.get(model_name, CLOUD_MODEL_CONFIGS["gpt-4o"])
        if not config:
            logger.error(f"No configuration found for model: {model_name}")
            raise ValueError(f"Invalid model name: {model_name}")
        return config

    def _init_openai_client(self) -> OpenAI:
        """
        Initialize OpenAI client with API key from either:
        1. Explicitly passed api_key parameter
        2. Environment variable OPENAI_API_KEY
        3. .env file
        """
        try:
            # Initialize client with config
            client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=self.model.get("api_base", "https://api.openai.com/v1"),
                timeout=self.model.get("timeout", 30),
                max_retries=self.model.get("max_retries", 2),
            )

            logger.info(f"Successfully initialized OpenAI client for model {self.name}")
            return client

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def process(
            self,
            messages: Iterable[ChatCompletionMessageParam],
            response_format: Union[completion_create_params.ResponseFormat, NotGiven],
            **kwargs,
    ) -> Dict:
        """
        Process input using the OpenAI client.

        Args:
            messages: List of messages to send to the model
            response_format: Format for the response (e.g., ExaminerResult)
        Returns:
            Dict containing the model response and metadata
        """
        logger.info(f"Using cloud model: {self.model['model']}")
        logger.info(f"Processing with configurations: {self.model}")

        try:
            # Create chat completion with proper error handling
            response = self.client.beta.chat.completions.parse(
                model=self.model["model"],
                messages=messages,
                response_format=response_format,
                max_tokens=self.model.get("max_tokens", 4096),
                temperature=self.model.get("temperature", 0.7),
                top_p=self.model.get("top_p", 1.0),
                presence_penalty=self.model.get("presence_penalty", 0),
                frequency_penalty=self.model.get("frequency_penalty", 0),
            )
            usage = response.usage.dict() if response.usage else None
            response = json.loads(response.choices[0].message.content)
            logger.info(f"Response from agent: {response}")
            return {
                "model": self.model["model"],
                "input": messages,
                "status": "processed",
                "response": response,
                "usage": usage,
            }

        except Exception as e:
            logger.error(f"Error processing input with OpenAI: {str(e)}")
            raise

    def process_realtime(self,
                         messages: Iterable[ChatCompletionMessageParam],
                         response_format: Union[completion_create_params.ResponseFormat, NotGiven],
                         **kwargs, ) -> Optional[str]:
        response = self.client.chat.completions.create(
            model="gpt-4o-realtime-preview-2024-12-17",
            modalities=["text", "audio"],
            messages=messages)
        return response.choices[0].message.content
