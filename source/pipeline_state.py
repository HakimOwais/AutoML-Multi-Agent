import os
import json
from typing import List, Any, Dict

from source.prompts.agent_prompts import *


class PipelineState:
    """
    Manages the workflow state including phase, memory (outputs per step),
    and provides methods for state persistence, context building, and saving the final state in Markdown.
    """
    def __init__(self, phase: str, output: str, message: str = "No message provided."):
        self.phase = phase
        self.output = output
        self.message = message
        self.memory: List[Dict[str, Any]] = [{}]
        self.current_step = 0
        self.score = 0
        self.finished = False

        # Define the agent prompt rules
        # (Make sure these constants are defined somewhere in your code)
        self.agent_rules = {
            "Agent Manager": AGENT_MANAGER_PROMPT,
            "Prompt Agent": PROMPT_AGENT_PROMPT,
            "Data Agent": DATA_AGENT_PROMPT,
            "Model Agent": MODEL_AGENT_PROMPT,
            "Operations Agent": OPERATION_AGENT_PROMPT
        }
        self.agents = list(self.agent_rules.keys())

        # Directories for saving state outputs
        self.output_dir = os.path.join("output", self.output)
        self.dir_name = self.phase.replace(" ", "_")
        self.restore_dir = os.path.join(self.output_dir, self.dir_name)
        self.context = ""

    def __str__(self) -> str:
        return (f"PipelineState(Phase: {self.phase}, Step: {self.current_step}, "
                f"Current Agent: {self.get_current_agent()}, Finished: {self.finished})")

    def make_context(self) -> None:
        """Build context information for the current phase."""
        self.context = f"Output: {self.output}\nPhase: {self.phase}\n"
        self.context += "Agents in workflow:\n"
        self.context += "\n".join(f"{i+1}. {agent}" for i, agent in enumerate(self.agents))

    def update_memory(self, memory_update: Dict[str, Any]) -> None:
        """Update internal memory with the current agent’s output."""
        print(f"[State] Updating memory for agent '{self.get_current_agent()}' (Phase: {self.phase}).")
        if self.memory and isinstance(self.memory[-1], dict):
            self.memory[-1].update(memory_update)
        else:
            self.memory.append(memory_update)

    def persist_memory(self) -> None:
        """Persist the memory to a JSON file for record keeping."""
        memory_path = os.path.join(self.restore_dir, "memory.json")
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        with open(memory_path, 'w') as f:
            json.dump(self.memory, f, indent=4)
        print(f"[State] Memory persisted to {memory_path}")

    def next_step(self) -> None:
        """Advance the workflow to the next step."""
        self.current_step += 1
        self.memory.append({})

    def get_current_agent(self) -> str:
        if self.agents:
            return self.agents[self.current_step % len(self.agents)]
        return "N/A"

    def make_dir(self) -> None:
        """Create necessary directories for state outputs."""
        os.makedirs(self.restore_dir, exist_ok=True)
