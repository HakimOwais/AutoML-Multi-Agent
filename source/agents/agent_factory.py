from source.agents.multi_agents import (AgentManager,
                          PromptAgent,
                          AutoMLAgent,
                          ModelAgent,
                          OperationsAgent)
from source.models.llm import groq_client
from source.schema import JSON_SCHEMA

groq_model_list = [
    "distil-whisper-large-v3-en",
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "whisper-large-v3",
    "whisper-large-v3-turbo"
]


agents = {
        "manager": AgentManager(
            client=groq_client,
            role="manager",
            model="llama-3.3-70b-versatile",
            description="Assistant project manager for parsing user requirements.",
            json_schema=JSON_SCHEMA,
            stream=False
        ),
        "prompt": PromptAgent(
            client=groq_client,
            role="prompt_parser",
            model="llama-3.3-70b-versatile",
            description="Assistant project manager for JSON parsing.",
            json_specification=JSON_SCHEMA,
            stream=False
        ),
        "automl": AutoMLAgent(
            client=groq_client,
            role="data_scientist",
            model="llama-3.3-70b-versatile",
            description="Automated ML agent for dataset preprocessing and augmentation.",
            data_path="data",
            stream=False
        ),
        "model": ModelAgent(
            client=groq_client,
            role="ml_researcher",
            model="llama-3.3-70b-versatile",
            description="ML research agent for model optimization and profiling.",
            stream=False
        ),
        "operations": OperationsAgent(
            client=groq_client,
            role="mlops",
            model="llama-3.3-70b-versatile",
            description="MLOps agent for model deployment.",
            stream=False
        )
    }