from groq import Groq
import os

from prompts.agent_prompts import agent_manager_prompt

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
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

# Manager Agent class
class AgentManager(AgentBase):
    def __init__(self, role, model, description, json_schema, **kwargs):
        super().__init__(role, model, description, **kwargs)
        self.json_schema = json_schema

    def parse_to_json(self, user_input):
        """Parses the user input into a JSON format based on the schema."""
        messages = [
            {
                "role": "system",
                "content": agent_manager_prompt,
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

# Example user input
user_input = "I need a task to analyze sales data with high priority. Deadline is next Friday. Allocate 2 data analysts and 1 machine learning engineer."

# Parse the input to JSON
json_response = manager_agent.parse_to_json(user_input)

# Print the JSON response
print(json_response)
