from groq import Groq
import os

from source.prompts.agent_prompts import (AGENT_MANAGER_PROMPT, 
                                   MODEL_AGENT_PROMPT, 
                                   PROMPT_AGENT_PROMPT,
                                   DATA_AGENT_PROMPT,
                                   OPERATION_AGENT_PROMPT)
from source.models.llm import ModelAgentBase
from source.schema import JSON_SCHEMA

# Manager Agent class
class AgentManager(ModelAgentBase):
    def __init__(self, client, role, model, description, json_schema, **kwargs):
        super().__init__(client, role, model, description, **kwargs)
        self.json_schema = json_schema

    async def parse_to_json(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": f"{AGENT_MANAGER_PROMPT}\n\n# JSON SPECIFICATION SCHEMA #\n{self.json_schema}"},
            {"role": "user", "content": user_input}
        ]
        response = await self.execute(messages)
        return response.choices[0].message.content
    
class PromptAgent(ModelAgentBase):
    def __init__(self, client, role, model, description, json_specification, **kwargs):
        super().__init__(client, role, model, description, **kwargs)
        self.json_specification = json_specification

    async def generate_json(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": PROMPT_AGENT_PROMPT.format(json_specification=self.json_specification)},
            {"role": "user", "content": user_input}
        ]
        response = await self.execute(messages)
        return response.choices[0].message.content

class AutoMLAgent(ModelAgentBase):
    def __init__(self, client, role, model, description, data_path: str = "./data", **kwargs):
        super().__init__(client, role, model, description, **kwargs)
        self.data_path = data_path

    async def preprocess_data(self, instructions: str) -> str:
        messages = [
            {"role": "system", "content": DATA_AGENT_PROMPT},
            {"role": "user", "content": f"Instructions: {instructions}"}
        ]
        response = await self.execute(messages)
        return response.choices[0].message.content
    
    async def augment_data(self, augmentation_details):
        """Perform augmentation if necessary"""
        messages = [
            {"role": "system", "content": DATA_AGENT_PROMPT},
            {"role": "user", "content": f"Augmentation Details: {augmentation_details}"},
        ]
        response = await self.execute(messages)
        return response.choices[0].message.content
    
    async def visualize_data(self, visualization_request):
        """Generate visualizations to understand the dataset"""
        messages = [
            {"role": "system", "content":DATA_AGENT_PROMPT},
            {"role": "user", "content": f"Visualization details {visualization_request}"},

        ]
        response = await self.execute(messages)
        return response.choices[0].message.content

class ModelAgent(ModelAgentBase):
    async def retrieve_models(self, dataset_details: str) -> str:
        messages = [
            {"role": "system", "content": MODEL_AGENT_PROMPT},
            {"role": "user", "content": dataset_details}
        ]
        response = await self.execute(messages)
        return response.choices[0].message.content

    async def optimize_model(self, hyperparameter_details: str) -> str:
        messages = [
            {"role": "system", "content": MODEL_AGENT_PROMPT},
            {"role": "user", "content": hyperparameter_details}
        ]
        response = await self.execute(messages)
        return response.choices[0].message.content

    async def profile_models(self, profiling_details: str) -> str:
        messages = [
            {"role": "system", "content": MODEL_AGENT_PROMPT},
            {"role": "user", "content": profiling_details}
        ]
        response = await self.execute(messages)
        return response.choices[0].message.content

class OperationsAgent(ModelAgentBase):
    async def deploy_model(self, deployment_details: str) -> str:
        messages = [
            {"role": "system", "content": OPERATION_AGENT_PROMPT},
            {"role": "user", "content": deployment_details}
        ]
        response = await self.execute(messages)
        return response.choices[0].message.content
