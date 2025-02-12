import os
from typing import Dict
from source.agents.multi_agents import ModelAgentBase
from source.pipeline_state import PipelineState
from source.embedding_manager import CSVEmbedder

# ============================
# COMPONENT: Pipeline Agent
# ============================
class PipelineAgent:
    """
    Orchestrates the entire pipeline from CSV ingestion through model deployment.
    Instead of returning separate outputs for each step, it aggregates all
    outputs into a single final result.
    """
    def __init__(self, agents: Dict[str, ModelAgentBase], state: PipelineState, csv_embedder: CSVEmbedder, dataset_dir: str = "data", **kwargs):
        self.agents = agents
        self.state = state
        self.csv_embedder = csv_embedder
        self.dataset_dir = dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.state.make_dir()

    async def run_pipeline(self, preprocessing_input: str, model_request: str, deployment_details: str) -> str:
        try:
            # --- Step 0: Query CSV embedder for context ---
            print("[Pipeline] Querying embedded CSV data for context...")
            memory_results = await self.csv_embedder.query_collection(preprocessing_input, n_results=5)
            memory_context = "\n".join(str(doc) for doc in memory_results.get("documents", []))
            print("[Pipeline] Retrieved memory context:\n", memory_context)

            # --- Step 1: Data Preprocessing via AutoMLAgent ---
            combined_preprocessing_input = f"{preprocessing_input}\n\nRelevant dataset context:\n{memory_context}"
            preprocessed_data = await self.agents["automl"].preprocess_data(combined_preprocessing_input)
            preprocessed_path = os.path.join(self.dataset_dir, "preprocessed_data.md")
            with open(preprocessed_path, "w") as f:
                f.write(preprocessed_data)
            print(f"[Pipeline] Preprocessed data saved to: {preprocessed_path}")
            self.state.update_memory({"preprocessing": preprocessed_data})
            self.state.persist_memory()
            self.state.next_step()

            # --- Step 2: Model Retrieval via ModelAgent ---
            combined_model_request = f"{model_request}\n\nRelevant dataset context:\n{memory_context}"
            model_list = await self.agents["model"].retrieve_models(combined_model_request)
            model_list_path = os.path.join(self.dataset_dir, "model_list.md")
            with open(model_list_path, "w") as f:
                f.write(model_list)
            print(f"[Pipeline] Model list saved to: {model_list_path}")
            self.state.update_memory({"model_list": model_list})
            self.state.persist_memory()
            self.state.next_step()

            # --- Step 3: Model Deployment via OperationsAgent ---
            combined_deployment_details = f"{deployment_details}\n\nRelevant dataset context:\n{memory_context}"
            deployment_output = await self.agents["operations"].deploy_model(combined_deployment_details)
            deployment_output_path = os.path.join(self.dataset_dir, "deployment_output.md")
            with open(deployment_output_path, "w") as f:
                f.write(deployment_output)
            print(f"[Pipeline] Deployment output saved to: {deployment_output_path}")
            self.state.update_memory({"deployment_output": deployment_output})
            self.state.persist_memory()
            self.state.next_step()

            # --- Combine all outputs into a single aggregated result ---
            final_output = (
                "Full Pipeline Result:\n\n"
                "===== Preprocessed Data =====\n" + preprocessed_data + "\n\n"
                "===== Model List =====\n" + model_list + "\n\n"
                "===== Deployment Output =====\n" + deployment_output
            )
            return final_output
        except Exception as e:
            error_message = f"An error occurred in the pipeline: {e}"
            print("[Pipeline]", error_message)
            self.state.update_memory({"error": error_message})
            self.state.persist_memory()
            raise e