import json
import asyncio
from source.pipeline_state import PipelineState
from source.embedding_manager import embedding_model, CSVEmbedder
from source.schema import JSON_SCHEMA
from source.agents.agent_factory import *
from source.agents.pipeline_agent import PipelineAgent

async def main():
    state = PipelineState(phase="Model Development", output="MyOutput")
    state.make_context()

    # embedding_model = embedding_model
    csv_embedder = CSVEmbedder(
        collection_name="auto_ml_memory",
        db_path="data/chromadb",
        embedding_model=embedding_model
    )
    # Optionally, embed CSV data (uncomment if needed):
    await csv_embedder.embed_csv("data/heart.csv")

    # JSON_SCHEMA = JSON_SCHEMA
    # agents = agents

    # --- Initialize and run the Pipeline Agent ---
    pipeline = PipelineAgent(agents=agents, state=state, csv_embedder=csv_embedder, dataset_dir="data")

    # --- Define sample inputs ---
    preprocessing_input = (
        "I have uploaded the dataset obtained from UCI Machine learning, "
        "which relates to detect the heart disease of patients based on various features of the patients." 
        "Develop a model with at least 90 percent accuracy. "
    )
    model_request = "Find the top 3 models for classifying this dataset."
    deployment_details = "Deploy the selected model as a web application."

    # --- Run the pipeline ---
    results = await pipeline.run_pipeline(preprocessing_input, model_request, deployment_details)
    print("Pipeline execution completed. Results:")
    print(json.dumps(results, indent=4))


    await csv_embedder.delete_collection()



if __name__ == "__main__":
    asyncio.run(main())