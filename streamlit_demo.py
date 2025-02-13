import os
import json
import asyncio
import streamlit_demo as st

# Import your pipeline modules
from source.pipeline_state import PipelineState
from source.embedding_manager import embedding_model, CSVEmbedder
from source.schema import JSON_SCHEMA
from source.agents.agent_factory import *
from source.agents.pipeline_agent import PipelineAgent

async def run_pipeline(preprocessing_input, model_request, deployment_details, dataset_path):
    """
    Runs the ML pipeline asynchronously using the user-provided inputs and dataset.
    """
    # --- Initialize Pipeline State ---
    state = PipelineState(phase="Model Development", output="MyOutput")
    state.make_context()

    # --- Initialize CSV Embedder ---
    csv_embedder = CSVEmbedder(
        collection_name="auto_ml_memory",
        db_path="data/chromadb",
        embedding_model=embedding_model
    )
    
    # Embed the uploaded CSV data.
    await csv_embedder.embed_csv(dataset_path)

    # --- Initialize the Pipeline Agent ---
    pipeline = PipelineAgent(
        agents=agents,       # agents imported from agent_factory
        state=state, 
        csv_embedder=csv_embedder, 
        dataset_dir="data"
    )

    # --- Run the pipeline ---
    results = await pipeline.run_pipeline(preprocessing_input, model_request, deployment_details)

    # --- Clean up ---
    await csv_embedder.delete_collection()

    return results

def main():
    st.title("ML Pipeline Dashboard")
    st.write("Enter the details below and upload your dataset to run the ML pipeline.")

    # --- Input boxes for pipeline parameters ---
    preprocessing_input = st.text_area(
        "Preprocessing Input", 
        "I have uploaded the dataset obtained from UCI Machine learning, which relates to detect the heart disease of patients based on various features of the patients. Develop a model with at least 90 percent accuracy."
    )
    model_request = st.text_input(
        "Model Request", 
        "Find the top 3 models for classifying this dataset."
    )
    deployment_details = st.text_input(
        "Deployment Details", 
        "Deploy the selected model as a web application."
    )

    # --- File uploader for the dataset CSV ---
    uploaded_file = st.file_uploader("Upload your dataset CSV", type=["csv"])

    if st.button("Run Pipeline"):
        if uploaded_file is None:
            st.error("Please upload a dataset CSV file.")
        else:
            # Ensure the data directory exists
            os.makedirs("data", exist_ok=True)
            # Save the uploaded CSV to a local file
            dataset_path = os.path.join("data", "uploaded_dataset.csv")
            with open(dataset_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.info("Dataset uploaded successfully. Running the pipeline...")

            # Run the asynchronous pipeline
            with st.spinner("Executing pipeline. Please wait..."):
                results = asyncio.run(
                    run_pipeline(
                        preprocessing_input, 
                        model_request, 
                        deployment_details, 
                        dataset_path
                    )
                )

            st.success("Pipeline execution completed!")
            st.subheader("Pipeline Results")
            st.json(results)

# if __name__ == "__main__":
#     main()
