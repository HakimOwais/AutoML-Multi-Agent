import os
from groq import Groq
import sys
from dotenv import load_dotenv

sys.path.insert(1, "source")


# Path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")

load_dotenv(dotenv_path)

from prompts.agent_prompts import (agent_manager_prompt,
                                   data_agent_prompt,
                                   model_agent_prompt,
                                   prompt_agent,
                                   operation_agent_prompt)

# Initialize Groq client
client = Groq(
    # api_key=os.environ.get("GROQ_API_KEY"),
    api_key="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"
)

# Base Agent class
class AgentBase:
    def __init__(self, role, model, description, **kwargs):
        self.role = role
        self.model = model
        self.description = description
        self.kwargs = kwargs

    def execute(self, messages):
        """Executes a task using the defined role and model."""
        return client.chat.completions.create(
            messages=messages,
            model=self.model,
            **self.kwargs
        )

# Agent Manager class
class AgentManager(AgentBase):
    def __init__(self, role, model, description, json_schema, **kwargs):
        super().__init__(role, model, description, **kwargs)
        self.json_schema = json_schema

    def parse_to_json(self, user_input):
        """Parses the user input into a JSON format based on the schema."""
        messages = [
            {
                "role": "system",
                "content": f"""
                {agent_manager_prompt.strip()}

                # JSON SPECIFICATION SCHEMA #
                {self.json_schema}
                """,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]
        response = self.execute(messages)
        return response.choices[0].message.content

# Define JSON specification schema
JSON_SCHEMA = """json
{
    "task": "string",
    "priority": "string",
    "deadline": "string",
    "resources": [
        {
            "type": "string",
            "quantity": "integer"
        }
    ]
}
"""

# Create the manager agent
manager_agent = AgentManager(
    role="manager",
    model="llama-3.3-70b-versatile",
    description="Assistant project manager for parsing user requirements into JSON.",
    json_schema=JSON_SCHEMA,
    stream=False
)

# Prompt Agent class
class PromptAgent(AgentBase):
    def __init__(self, role, model, description, json_specification, **kwargs):
        super().__init__(role, model, description, **kwargs)
        self.json_specification = json_specification

    def generate_json(self, user_input):
        """Generates a JSON response strictly adhering to the specification."""
        messages = [
            {
                "role": "system",
                "content": f"""
                {prompt_agent.strip()}

                # JSON SPECIFICATION SCHEMA #
                '''json
                {self.json_specification}
                '''
                """,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]
        response = self.execute(messages)
        return response.choices[0].message.content

# Create the prompt agent
prompt_agent = PromptAgent(
    role="prompt_parser",
    model="llama-3.3-70b-versatile",
    description="Assistant project manager for JSON parsing.",
    json_specification=JSON_SCHEMA,
    stream=False
)

class AutoMLAgent(AgentBase):
    def __init__(self, role, model, description, data_path="./data", **kwargs):
        super().__init__(role, model, description, **kwargs)
        self.data_path = data_path

    def retrieve_dataset(self, query):
        """Retrieves a dataset based on user instructions or searches for one."""
        dataset_path = os.path.join(self.data_path, "renttherunway_cleaned.csv")
        messages = [
            {
                "role": "system",
                "content": f"""
                {data_agent_prompt.strip()}
                """,
            },
            {
                "role": "user",
                "content": query,
            },
        ]
        response = self.execute(messages)
        # Save the retrieved dataset to the specified path (placeholder implementation)
        with open(dataset_path, "w") as file:
            file.write(response.choices[0].message.content)
        return dataset_path

    def preprocess_data(self, instructions):
        """Performs data preprocessing based on user instructions or best practices."""
        messages = [
            {
                "role": "system",
                "content": f"""
                {data_agent_prompt.strip()}
                """,
            },
            {
                "role": "user",
                "content": f"Instructions: {instructions}",
            },
        ]
        response = self.execute(messages)
        return response.choices[0].message.content

    def augment_data(self, augmentation_details):
        """Performs data augmentation as necessary."""
        messages = [
            {
                "role": "system",
                "content": f"""
                {data_agent_prompt.strip()}
                """,
            },
            {
                "role": "user",
                "content": f"Augmentation Details: {augmentation_details}",
            },
        ]
        response = self.execute(messages)
        return response.choices[0].message.content

    def visualize_data(self, visualization_request):
        """Generates meaningful visualizations to understand the dataset."""
        messages = [
            {
                "role": "system",
                "content": f"""
                {data_agent_prompt.strip()}
                """,
            },
            {
                "role": "user",
                "content": visualization_request,
            },
        ]
        response = self.execute(messages)
        return response.choices[0].message.content

# Create the AutoML agent
automl_agent = AutoMLAgent(
    role="data_scientist",
    model="llama-3.3-70b-versatile",
    description="Automated machine learning agent for dataset retrieval, preprocessing, augmentation, and visualization.",
    stream=False
)

# Model Agent class
class ModelAgent(AgentBase):
    def __init__(self, role, model, description, **kwargs):
        super().__init__(role, model, description, **kwargs)

    def retrieve_models(self, dataset_details):
        """Retrieve a list of well-performing models or algorithms based on dataset details."""
        messages = [
            {
                "role": "system",
                "content": f"""
                {model_agent_prompt.strip()}
                """,
            },
            {
                "role": "user",
                "content": dataset_details,
            },
        ]
        response = self.execute(messages)
        return response.choices[0].message.content

    def optimize_model(self, hyperparameter_details):
        """Perform hyperparameter optimization on candidate models."""
        messages = [
            {
                "role": "system",
                "content": f"""
                {model_agent_prompt.strip()}
                """,
            },
            {
                "role": "user",
                "content": hyperparameter_details,
            },
        ]
        response = self.execute(messages)
        return response.choices[0].message.content

    def profile_models(self, profiling_details):
        """Perform metadata extraction and profiling on candidate models."""
        messages = [
            {
                "role": "system",
                "content": f"""
                {model_agent_prompt.strip()}
                """,
            },
            {
                "role": "user",
                "content": profiling_details,
            },
        ]
        response = self.execute(messages)
        return response.choices[0].message.content

# Create the Model agent
model_agent = ModelAgent(
    role="ml_researcher",
    model="llama-3.3-70b-versatile",
    description="Machine learning research agent for model optimization and profiling.",
    stream=False
)

# Operations Agent class
class OperationsAgent(AgentBase):
    def __init__(self, role, model, description, **kwargs):
        super().__init__(role, model, description, **kwargs)

    def deploy_model(self, deployment_details):
        """Prepare and deploy the model based on the provided details."""
        messages = [
            {
                "role": "system",
                "content": f"""
                {operation_agent_prompt.strip()}
                """,
            },
            {
                "role": "user",
                "content": deployment_details,
            },
        ]
        response = self.execute(messages)
        return response.choices[0].message.content

# Create the Operations agent
operations_agent = OperationsAgent(
    role="mlops",
    model="llama-3.3-70b-versatile",
    description="MLOps agent for deployment and application development.",
    stream=False
)

# Sync all agents
agents = {
    "manager": manager_agent,
    "prompt": prompt_agent,
    "automl": automl_agent,
    "model": model_agent,
    "operations": operations_agent
}

# Preprocess data
user_input = "I have uploaded the dataset which is obtained from rent the runway and this dataset is related to fit fiber clothing for women. Develop a model with atleast 90 percent of F1 score. Also its target variable is fit"
preprocessed_data = automl_agent.preprocess_data(user_input)
with open("data/preprocessed_data.txt", "w") as f:
    f.write(preprocessed_data)

# Retrieve models
model_request = "Find the top 3 models for classifying this dataset."
model_list = model_agent.retrieve_models(model_request)
with open("data/model_list.txt", "w") as f:
    f.write(model_list)

# Deploy the model
deployment_details = "Deploy the selected model as a web application."
deployment_output = operations_agent.deploy_model(deployment_details)
with open("data/deployment_output.txt", "w") as f:
    f.write(deployment_output)

# Example usage
# user_input = "I need to preprocess a dataset for anomaly detection in financial transactions."
# preprocessed_data = automl_agent.preprocess_data(user_input)
# print(preprocessed_data)

# model_request = "Find the top 3 models for classifying this dataset."
# model_list = model_agent.retrieve_models(model_request)
# print(model_list)

# deployment_details = "Deploy the selected model as a web application."
# deployment_output = operations_agent.deploy_model(deployment_details)
# print(deployment_output)

# The dataset is being accessed in the following methods:
# - "retrieve_dataset" in the AutoMLAgent class.
# - "preprocess_data" in the AutoMLAgent class.
# - "augment_data" in the AutoMLAgent class.
# - "visualize_data" in the AutoMLAgent class.

# Suggested placement for train and test sets
# The train and test sets should be stored in a secure location, preferably a dedicated directory within the project structure. For example:
# ./data/train/ - for training data
# ./data/test/ - for testing data
# Always use relative paths or environment variables to reference these directories to ensure compatibility across environments.
