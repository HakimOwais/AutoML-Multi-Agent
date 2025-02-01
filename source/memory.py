import os
import json
import pandas as pd
import chromadb
from chromadb.config import Settings

class CSVEmbeddingManager:
    """
    CSVEmbeddingManager handles the ingestion of CSV data into a Chroma DB
    collection and supports continuous updates and queries. This is useful
    for storing intermediate outputs (e.g., from data ingestion or model development)
    for later retrieval.
    """
    def __init__(self, collection_name="default_collection", db_path="chromadb", cache_size=10_000_000_000):
        self.settings = Settings(
            chroma_segment_cache_policy="LRU",
            chroma_memory_limit_bytes=cache_size  # ~10GB
        )
        # Initialize a persistent Chroma DB client.
        self.client = chromadb.PersistentClient(path=db_path, settings=self.settings)
        self.collection = self.client.get_or_create_collection(collection_name)

    def reset_collection(self) -> None:
        """
        Resets the collection by clearing existing data.
        """
        self.client.reset()
        self.collection = self.client.get_or_create_collection(self.collection.name)

    def embed_csv(self, csv_file_path: str, batch_size: int = 1000) -> None:
        """
        Embeds the CSV data into the collection.
        Args:
            csv_file_path (str): Path to the CSV file.
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        df = pd.read_csv(csv_file_path)

        # Ensure there is an 'id' column for unique identification.
        if 'id' not in df.columns:
            df['id'] = df.index.astype(str)

        # Calculate the number of batches
        num_batches = (len(df) // batch_size) + int(len(df) % batch_size > 0)
        for i in range(num_batches):
            batch_df = df[i * batch_size:(i+1) * batch_size]
            ids = batch_df['id'].astype(str).tolist()
            documents = batch_df.drop(columns=['id'], errors='ignore').apply(lambda row: row.to_json(), axis=1).tolist()
            metadatas = batch_df.drop(columns=['id'], errors='ignore').to_dict(orient='records')

            # Upsert data into the collection.
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"Batch {i+1}/{num_batches} embedded successfully.")

    def update_embedding(self, csv_file_path: str) -> None:
        """
        Continuously update the collection with new or modified CSV data.
        Args:
            csv_file_path (str): Path to the CSV file to update.
        """
        print(f"[Memory] Updating embedding with data from {csv_file_path}.")
        self.embed_csv(csv_file_path)

    def query_collection(self, query_texts: list, where_clause: dict = None) -> dict:
        """
        Queries the collection for matching documents.
        Args:
            query_texts (list): List of query strings.
            where_clause (dict, optional): Additional filters for the query.
        Returns:
            dict: Query results.
        """
        result = self.collection.query(query_texts=query_texts, where_document=where_clause or {})
        return result

    def save_query_results(self, query_results: dict, output_path: str = "query_results.json") -> None:
        """
        Saves query results to a JSON file.
        Args:
            query_results (dict): Results from the query.
            output_path (str): Path to save the JSON file.
        """
        with open(output_path, "w") as file:
            json.dump(query_results, file, indent=4)
        print(f"[Memory] Query results saved to {output_path}")
