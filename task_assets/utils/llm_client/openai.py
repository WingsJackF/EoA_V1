import logging
from typing import Optional
from .base import BaseClient

try:
    from implement_llm_interaction_module.env_loader import get_env
except ImportError:
    get_env = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = 'openai'


logger = logging.getLogger(__name__)

class OpenAIClient(BaseClient):

    ClientClass = OpenAI

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(model, temperature)
        
        if isinstance(self.ClientClass, str):
            logger.fatal(f"Package `{self.ClientClass}` is required")
            exit(-1)

        if get_env is not None:
            api_key = api_key or get_env("LLM_API_KEY", aliases=("OPENAI_API_KEY",))
            base_url = base_url or get_env("LLM_BASE_URL", aliases=("OPENAI_BASE_URL",))

        self.client = self.ClientClass(api_key=api_key, base_url=base_url)
    
    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, n=n, stream=False,
        )
        return response.choices
