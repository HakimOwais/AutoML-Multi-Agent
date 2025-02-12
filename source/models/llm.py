import os
from groq import Groq
from dotenv import load_dotenv
import asyncio
from typing import Any, Dict, List

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

class ModelAgentBase:
    """
    Provides a common asynchronous interface for all agents.
    The actual call to the groq client
    """
    def __init__(self, client, role:str, model:str, 
                 description: str, **kwargs):
        self.client = client
        self.role = role
        self.model = model
        self.description = description
        self.kwargs = kwargs

    async def execute(self, messages: List[Dict[str,str]]) -> Any:
        return await asyncio.to_thread(
            self.client.chat.completions.create,
            messages = messages,
            model=self.model,
            **self.kwargs
        )
