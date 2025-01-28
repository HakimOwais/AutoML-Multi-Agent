# # from chromadb.config import Settings
# from chromadb.config import Settings

# settings = Settings(
#     chroma_segment_cache_policy="LRU",
#     chroma_memory_limit_bytes=10000000000  # ~10GB
# )




# import chromadb
# from chromadb.config import Settings

# client = chromadb.PersistentClient(path="test", settings=Settings(allow_reset=True))

# client.reset()
# col = client.get_or_create_collection("test")

# col.upsert(ids=["1", "2", "3"], documents=["He is a technology freak and he loves AI topics", "AI technology are advancing at a fast pace", "Innovation in LLMs is a hot topic"],metadatas=[{"author": "John Doe"}, {"author": "Jane Doe"}, {"author": "John Doe"}])
# print(col.query(query_texts=["technology"], where_document={"$or":[{"$contains":"technology"}, {"$contains":"freak"}]}))

import os
import json
import pandas as pd
import chromadb
from chromadb.config import Settings

class CSVEmbedding:
    def __init__(self, collection_name="default_collection", db_path="chromadb", cache_size=10_000_000_000):
        """Initializes the CSVEmbedding class."""
        self.settings = Settings(
            chroma_segment_cache_policy="LRU",
            chroma_memory_limit_bytes=cache_size  # ~10GB
        )
        self.client = chromadb.PersistentClient(path=db_path, settings=self.settings)
        self.collection = self.client.get_or_create_collection(collection_name)

    def reset_collection(self):
        """Resets the collection."""
        self.client.reset()
        self.collection = self.client.get_or_create_collection(self.collection.name)

    def embed_csv(self, csv_file_path):
        """
        Embeds the CSV data into the collection.
        Args:
            csv_file_path (str): Path to the CSV file.
        """
        # Load CSV file
        df = pd.read_csv(csv_file_path)

        # If 'id' column does not exist, create an 'id' column from the index
        if 'id' not in df.columns:
            df['id'] = df.index.astype(str)

        # Prepare data for embedding
        ids = df['id'].astype(str).tolist()
        documents = df.drop(columns=['id'], errors='ignore').apply(lambda row: row.to_json(), axis=1).tolist()
        metadatas = df.drop(columns=['id'], errors='ignore').to_dict(orient='records')
        
        # Upsert into collection
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query_collection(self, query_texts, where_clause=None):
        """
        Queries the collection.
        Args:
            query_texts (list): List of query strings.
            where_clause (dict): Optional query filters.
        Returns:
            dict: Query results.
        """
        return self.collection.query(query_texts=query_texts, where_document=where_clause or {})

    def save_query_results(self, query_results, output_path="query_results.json"):
        """
        Saves query results to a JSON file.
        Args:
            query_results (dict): Results of the query.
            output_path (str): File path to save the results.
        """
        with open(output_path, "w") as file:
            json.dump(query_results, file, indent=4)


# from csv_embedding import CSVEmbedding  # Assuming you save the above class in csv_embedding.py

class AutoMLAgent(AgentBase):
    def __init__(self, role, model, description, data_path="./data", **kwargs):
        super().__init__(role, model, description, **kwargs)
        self.data_path = data_path
        self.embedding_manager = CSVEmbedding(collection_name="multi_agent_workflow")

    def embed_dataset(self, csv_file):
        """
        Embeds a dataset into the ChromaDB collection.
        Args:
            csv_file (str): Path to the CSV file to embed.
        """
        self.embedding_manager.embed_csv(csv_file)

    def retrieve_dataset(self, query):
        """Retrieves a dataset based on user instructions or searches for one."""
        dataset_path = os.path.join(self.data_path, "renttherunway_cleaned.csv")
        messages = [
            {
                "role": "system",
                "content": f"""
                {data_agent_prompt.strip()}
                """,
            },
            {
                "role": "user",
                "content": query,
            },
        ]
        response = self.execute(messages)
        with open(dataset_path, "w") as file:
            file.write(response.choices[0].message.content)
        return dataset_path

    def query_embedded_data(self, query_texts, where_clause=None):
        """
        Queries embedded data using the CSVEmbedding class.
        Args:
            query_texts (list): List of query strings.
            where_clause (dict): Optional filters for the query.
        Returns:
            dict: Query results.
        """
        return self.embedding_manager.query_collection(query_texts, where_clause)

    def preprocess_data(self, instructions):
        """Performs data preprocessing based on user instructions or best practices."""
        messages = [
            {
                "role": "system",
                "content": f"""
                {data_agent_prompt.strip()}
                """,
            },
            {
                "role": "user",
                "content": f"Instructions: {instructions}",
            },
        ]
        response = self.execute(messages)
        return response.choices[0].message.content

    def augment_data(self, augmentation_details):
        """Performs data augmentation as necessary."""
        messages = [
            {
                "role": "system",
                "content": f"""
                {data_agent_prompt.strip()}
                """,
            },
            {
                "role": "user",
                "content": f"Augmentation Details: {augmentation_details}",
            },
        ]
        response = self.execute(messages)
        return response.choices[0].message.content

    def visualize_data(self, visualization_request):
        """Generates meaningful visualizations to understand the dataset."""
        messages = [
            {
                "role": "system",
                "content": f"""
                {data_agent_prompt.strip()}
                """,
            },
            {
                "role": "user",
                "content": visualization_request,
            },
        ]
        response = self.execute(messages)
        return response.choices[0].message.content

