import os
import sys
import json
from typing import List, Dict, Any, Optional

# Add current directory to path (if needed)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Agent Prompts as Rules ---
agent_manager_prompt = """
You are an experienced senior project manager of an automated machine learning project (AutoML).
You have two main responsibilities as follows.
1. Receive requirements and/or inquiries from users through a well-structured JSON object.
2. Using recent knowledge and state-of-the-art studies to devise promising high-quality
plans for data scientists, machine learning research engineers, and MLOps engineers in
your team to execute subsequent processes based on the user requirements you have
received.
""".strip()

prompt_agent = """
You are an assistant project manager in the AutoML development team.
Your task is to parse the user's requirement into a valid JSON format using the JSON
specification schema as your reference. Your response must exactly follow the given
JSON schema and be based only on the user's instruction.
Make sure that your answer contains only the JSON response without any comment or
explanation because it can cause parsing errors.
#JSON SPECIFICATION SCHEMA#
'''json
{json_specification}
'''
Your response must begin with "'''json" or "{{" and end with "'''" or "}}", respectively.
""".strip()

data_agent_prompt = """
You are the world's best data scientist of an automated machine learning project (AutoML)
that can find the most relevant datasets, run useful preprocessing, perform suitable
data augmentation, and make meaningful visualizations to comprehensively understand the
data based on the user requirements. You have the following main responsibilities to
complete.
1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
2. Perform data preprocessing based on the user instruction or best practices.
3. Perform data augmentation as necessary.
4. Extract useful information and underlying characteristics of the dataset.
""".strip()

model_agent_prompt = """
You are the world's best machine learning research engineer of an automated machine
learning project (AutoML) that can find the optimal candidate machine learning models
and artificial intelligence algorithms for the given dataset(s), run hyperparameter
tuning to optimize the models, and perform metadata extraction and profiling to
comprehensively understand the candidate models or algorithms based on the user
requirements. You have the following main responsibilities to complete.
1. Retrieve a list of well-performing candidate ML models and AI algorithms for the given
dataset based on the user's requirement and instruction.
2. Perform hyperparameter optimization for those candidate models or algorithms.
3. Extract useful information and underlying characteristics of the candidate models or
algorithms using metadata extraction and profiling techniques.
4. Select the top-k ('k' will be given) well-performing models or algorithms based on the
hyperparameter optimization and profiling results.
""".strip()

operation_agent_prompt = """
You are the world's best MLOps engineer of an automated machine learning project (AutoML)
that can implement the optimal solution for production-level deployment, given any
datasets and models. You have the following main responsibilities to complete.
1. Write accurate Python codes to retrieve/load the given dataset from the corresponding source.
2. Write effective Python codes to preprocess the retrieved dataset.
3. Write precise Python codes to retrieve/load the given model and optimize it with the
suggested hyperparameters.
4. Write efficient Python codes to train/finetune the retrieved model.
5. Write suitable Python codes to prepare the trained model for deployment. This step may
include model compression and conversion according to the target inference platform.
6. Write Python codes to build the web application demo using the Gradio library.
7. Run the model evaluation using the given Python functions and summarize the results for
validation against the user's requirements.
""".strip()

# --- End Agent Prompts ---

class State:
    """
    The State class tracks the current phase, the ordered list of agents (using your
    prompt rules as the definition of each agent's responsibilities), and the internal
    memory of outputs from each step.
    """
    def __init__(self, phase: str, competition: str, message: str = "No message provided."):
        self.phase = phase
        self.competition = competition
        self.message = message
        self.memory: List[Dict[str, Any]] = [{}]  # Stores outputs per step
        self.current_step = 0
        self.score = 0
        self.finished = False

        # Define the agent prompt rules as a dictionary.
        # You can adjust the keys or order as needed.
        self.agent_rules = {
            "Agent Manager": agent_manager_prompt,
            "Prompt Agent": prompt_agent,
            "Data Agent": data_agent_prompt,
            "Model Agent": model_agent_prompt,
            "Operations Agent": operation_agent_prompt
        }
        # Use the keys of the dictionary as the ordered list of agents.
        self.agents = list(self.agent_rules.keys())
        
        # Directories for saving state outputs.
        self.competition_dir = os.path.join("competition", self.competition)
        # Use a simple directory name based on phase (replace spaces with underscores)
        self.dir_name = self.phase.replace(" ", "_")
        self.restore_dir = os.path.join(self.competition_dir, self.dir_name)
        
        self.background_info = ""
        self.context = ""

    def __str__(self) -> str:
        return (f"State(Phase: {self.phase}, Step: {self.current_step}, "
                f"Current Agent: {self.get_current_agent()}, Finished: {self.finished})")

    def make_context(self) -> None:
        """
        Build context information using competition and phase details and list all agents.
        """
        self.context = f"Competition: {self.competition}\nPhase: {self.phase}\n"
        self.context += "Agents in workflow:\n"
        self.context += "\n".join(f"{i+1}. {agent}" for i, agent in enumerate(self.agents))
    
    def get_state_info(self) -> str:
        """
        Returns high-level instructions for the current phase by referencing the relevant agent rules.
        """
        if self.phase == "Data Ingestion":
            return (
                "Data Ingestion Phase:\n"
                "Responsibilities include:\n"
                f"- {self.agent_rules.get('Agent Manager')}\n"
                f"- {self.agent_rules.get('Data Agent')}\n"
                "Output: Raw data ingested and stored in the memory system (Chroma DB).\n"
            )
        elif self.phase == "Model Development":
            return (
                "Model Development Phase:\n"
                "Responsibilities include:\n"
                f"- {self.agent_rules.get('Model Agent')}\n"
                f"- {self.agent_rules.get('Operations Agent')}\n"
                "Output: Trained models and evaluation reports produced.\n"
            )
        else:
            # For any other phase, list all agents.
            return f"Phase: {self.phase} with agents: {', '.join(self.agents)}"

    def set_background_info(self, background_info: str) -> None:
        self.background_info = background_info

    def get_current_agent(self) -> str:
        if self.agents:
            return self.agents[self.current_step % len(self.agents)]
        return "N/A"

    def generate_rules(self) -> str:
        """
        Retrieve the rule text (agent prompt) for the current agent.
        Save it to a file for record keeping.
        """
        agent = self.get_current_agent()
        rules = self.agent_rules.get(agent, "No rules defined for this agent.")
        rules_path = os.path.join(self.restore_dir, "user_rules.txt")
        os.makedirs(os.path.dirname(rules_path), exist_ok=True)
        with open(rules_path, 'w') as f:
            f.write(rules)
        return rules

    def make_dir(self) -> None:
        """
        Create directories for the competition and current phase outputs.
        """
        os.makedirs(self.restore_dir, exist_ok=True)

    def update_memory(self, memory_update: Dict[str, Any]) -> None:
        """
        Update the internal memory with new output information from the current agent's task.
        """
        print(f"[State] Updating memory for agent '{self.get_current_agent()}' in Phase: {self.phase}.")
        if self.memory and isinstance(self.memory[-1], dict):
            self.memory[-1].update(memory_update)
        else:
            self.memory.append(memory_update)

    def persist_memory(self) -> None:
        """
        Persist the current memory to a JSON file.
        """
        memory_path = os.path.join(self.restore_dir, "memory.json")
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        with open(memory_path, 'w') as f:
            json.dump(self.memory, f, indent=4)
        print(f"[State] Memory persisted to {memory_path}")

    def restore_report(self) -> None:
        """
        Write out any report from the last memory update.
        """
        report = self.memory[-1].get("summarizer", {}).get("report", "")
        if report:
            report_path = os.path.join(self.restore_dir, "report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"[State] Report restored to {report_path}")
        else:
            print(f"[State] No report available for Phase: {self.phase}")

    def next_step(self) -> None:
        """
        Advance the workflow to the next agent/step.
        """
        self.current_step += 1
        self.memory.append({})

    def set_score(self) -> None:
        """
        Compute an average score based on evaluation results stored in memory.
        """
        final_score = self.memory[-1].get("reviewer", {}).get("score", {})
        if final_score.get("agent developer", 3) == 0:
            self.score = 0
        else:
            self.score = sum(float(score) for score in final_score.values()) / len(final_score)

    def check_finished(self) -> bool:
        """
        Determine if all agents in the workflow have executed.
        """
        self.finished = self.current_step >= len(self.agents)
        return self.finished
