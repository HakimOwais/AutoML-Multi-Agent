from groq import Groq
import os

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain the importance of fast language models",
#         }
#     ],
#     model="llama-3.3-70b-versatile",
#     stream=False,
# )

# print(chat_completion.choices[0].message.content)


# # Define custom agents
# class CustomAgent:
#     def __init__(self, role, model, **kwargs):
#         self.role = role
#         self.model = model
#         self.kwargs = kwargs

#     def execute(self, messages):
#         return client.chat.completions.create(
#             messages=messages,
#             model=self.model,
#             **self.kwargs
#         )

# # Create specific agents
# user_agent = CustomAgent(
#     role="user",
#     model="llama-3.3-70b-versatile",
#     stream=False
# )

# # Define the conversation
# messages = [
#     {
#         "role": "user",
#         "content": "Explain the importance of fast language models",
#     }
# ]

# # Execute using the user agent
# response = user_agent.execute(messages)

# # Print the result
# print(response.choices[0].message.content)
