import os
from typing import Dict
from source.agents.multi_agents import ModelAgentBase
from source.pipeline_state import PipelineState
from source.embedding_manager import CSVEmbedder

class PipelineAgent:
    """
    Orchestrates the entire pipeline from CSV ingestion through model deployment.
    This version uses all five agents to produce a single aggregated final output.
    Outputs from each agent are saved as separate files in the dataset directory.
    """
    def __init__(self, agents: Dict[str, ModelAgentBase], state: PipelineState, csv_embedder: CSVEmbedder, dataset_dir: str = "data", **kwargs):
        self.agents = agents
        self.state = state
        self.csv_embedder = csv_embedder
        self.dataset_dir = dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.state.make_dir()

    async def run_pipeline(self, preprocessing_input: str, model_request: str, deployment_details: str) -> dict:
        try:
            # --- Step 0: Use Manager Agent to parse initial requirements ---
            print("[Pipeline] Parsing user requirements using AgentManager...")
            manager_output = await self.agents["manager"].parse_to_json(preprocessing_input)
            manager_output_path = os.path.join(self.dataset_dir, "manager_output.md")
            with open(manager_output_path, "w") as f:
                f.write(manager_output)
            print(f"[Pipeline] Manager output saved to: {manager_output_path}")
            self.state.update_memory({"manager_output": manager_output})
            self.state.persist_memory()
            self.state.next_step()

            # --- Step 1: Use Prompt Agent to refine the requirements ---
            print("[Pipeline] Refining requirements using PromptAgent...")
            prompt_output = await self.agents["prompt"].generate_json(manager_output)
            prompt_output_path = os.path.join(self.dataset_dir, "prompt_output.md")
            with open(prompt_output_path, "w") as f:
                f.write(prompt_output)
            print(f"[Pipeline] Prompt output saved to: {prompt_output_path}")
            self.state.update_memory({"prompt_output": prompt_output})
            self.state.persist_memory()
            self.state.next_step()

            # --- Step 2: Query CSV embedder for context ---
            print("[Pipeline] Querying embedded CSV data for context...")
            memory_results = await self.csv_embedder.query_collection(preprocessing_input, n_results=5)
            memory_context = "\n".join(str(doc) for doc in memory_results.get("documents", []))
            csv_context_path = os.path.join(self.dataset_dir, "csv_context.md")
            with open(csv_context_path, "w") as f:
                f.write(memory_context)
            print(f"[Pipeline] CSV context saved to: {csv_context_path}")
            self.state.update_memory({"csv_context": memory_context})
            self.state.persist_memory()
            self.state.next_step()

            # --- Step 3: Data Preprocessing via AutoMLAgent ---
            # Combine the refined requirements with CSV context.
            combined_preprocessing_input = f"{prompt_output}\n\nRelevant dataset context:\n{memory_context}"
            preprocessed_data = await self.agents["automl"].preprocess_data(combined_preprocessing_input)
            preprocessed_path = os.path.join(self.dataset_dir, "preprocessed_data.md")
            with open(preprocessed_path, "w") as f:
                f.write(preprocessed_data)
            print(f"[Pipeline] Preprocessed data saved to: {preprocessed_path}")
            self.state.update_memory({"preprocessed_data": preprocessed_data})
            self.state.persist_memory()
            self.state.next_step()

            # --- Step 4: Model Retrieval via ModelAgent ---
            combined_model_request = f"{model_request}\n\nRelevant dataset context:\n{memory_context}"
            model_list = await self.agents["model"].retrieve_models(combined_model_request)
            model_list_path = os.path.join(self.dataset_dir, "model_list.md")
            with open(model_list_path, "w") as f:
                f.write(model_list)
            print(f"[Pipeline] Model list saved to: {model_list_path}")
            self.state.update_memory({"model_list": model_list})
            self.state.persist_memory()
            self.state.next_step()

            # --- Step 5: Model Deployment via OperationsAgent ---
            combined_deployment_details = f"{deployment_details}\n\nRelevant dataset context:\n{memory_context}"
            deployment_output = await self.agents["operations"].deploy_model(combined_deployment_details)
            deployment_output_path = os.path.join(self.dataset_dir, "deployment_output.md")
            with open(deployment_output_path, "w") as f:
                f.write(deployment_output)
            print(f"[Pipeline] Deployment output saved to: {deployment_output_path}")
            self.state.update_memory({"deployment_output": deployment_output})
            self.state.persist_memory()
            self.state.next_step()

            # --- Aggregate all outputs into a single final result ---
            final_result = {
                "manager_output": manager_output,
                "prompt_output": prompt_output,
                "csv_context": memory_context,
                "preprocessed_data": preprocessed_data,
                "model_list": model_list,
                "deployment_output": deployment_output,
                "aggregated_output": (
                    "Full Pipeline Result:\n\n"
                    "===== Manager Output =====\n" + manager_output + "\n\n" +
                    "===== Prompt Output =====\n" + prompt_output + "\n\n" +
                    "===== CSV Context =====\n" + memory_context + "\n\n" +
                    "===== Preprocessed Data =====\n" + preprocessed_data + "\n\n" +
                    "===== Model List =====\n" + model_list + "\n\n" +
                    "===== Deployment Output =====\n" + deployment_output
                )
            }
            return final_result

        except Exception as e:
            error_message = f"An error occurred in the pipeline: {e}"
            print("[Pipeline]", error_message)
            self.state.update_memory({"error": error_message})
            self.state.persist_memory()
            raise e
