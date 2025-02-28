import logging
from typing import Any, Dict

from omni.llm.base import BaseAgent
from omni.llm.cloud import CloudAgent
from omni.llm.exceptions import InvalidAgentType
from omni.llm.ollama import OllamaAgent
from omni.llm.realtime import RealtimeAgent

logger = logging.getLogger(__name__)


class AgentManager:
    def __init__(self, agent_name: str, agent_type: str, **kwargs):
        """
        Initialize AgentManager with a specific agent.

        Args:
            agent_name: Name for the agent (e.g., 'gpt-4', 'gpt-4-turbo')
            agent_type: Type of agent ('cloud', 'quantization', or 'hf')
        """
        self.agent_types = {
            "cloud": CloudAgent,
            "ollama": OllamaAgent,
            "realtime": RealtimeAgent,
        }

        self.agent = self._init_agent(agent_name, agent_type, **kwargs)

    def _init_agent(self, agent_name: str, agent_type: str, **kwargs) -> BaseAgent:
        agent_type = agent_type.lower()
        if agent_type not in self.agent_types:
            raise InvalidAgentType(
                f"Invalid agent type. Must be one of: {', '.join(self.agent_types.keys())}"
            )

        agent_class = self.agent_types[agent_type]
        return agent_class(agent_name, agent_type, **kwargs)

    def process(
        self,
        **kwargs,
    ) -> Any:
        return self.agent.process(**kwargs)

    def get_agent_info(self) -> Dict[str, str]:
        return {
            "name": self.agent.name,
            "type": type(self.agent).__name__,
            "config": getattr(self.agent, "model", None),
        }
