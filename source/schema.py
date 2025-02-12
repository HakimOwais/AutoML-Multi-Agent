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