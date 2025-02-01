# import re


# def loop(max_iterations=10, query: str = ""):

#     agent = Agent(client=client, system=system_prompt)

#     tools = ["", ""]

#     next_prompt = query

#     i = 0
  
#     while i < max_iterations:
#         i += 1
#         result = agent(next_prompt)
#         print(result)

#         if "PAUSE" in result and "Action" in result:
#             action = re.findall(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)
#             chosen_tool = action[0][0]
#             arg = action[0][1]

#             if chosen_tool in tools:
#                 result_tool = eval(f"{chosen_tool}('{arg}')")
#                 next_prompt = f"Observation: {result_tool}"

#             else:
#                 next_prompt = "Observation: Tool not found"

#             print(next_prompt)
#             continue

#         if "Answer" in result:
#             break


# loop(query="What is the mass of Earth plus the mass of Saturn and all of that times 2?")