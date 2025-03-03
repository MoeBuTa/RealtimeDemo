
from typing import Any, Dict

from loguru import logger
import yaml

from realtime.utils.constants import CONFIG_FILE


def load_config(config_path) -> Dict[str, Any]:
    try:
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file)
    except FileNotFoundError:
        logger.warning(
            f"Config file not found at {config_path}. Using default configuration."
        )
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        return {}


config = load_config(CONFIG_FILE)

OPENAI_API_KEY = config["openai"]["api_key"]


REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
HEADERS = [
    f"Authorization: Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta: realtime=v1"
]

# Audio configuration
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate for recording (16kHz)
MAX_RECORD_SECONDS = 30  # Maximum recording time