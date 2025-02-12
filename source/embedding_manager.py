import os
import chromadb
import asyncio
from tqdm import tqdm
from chromadb.config import Settings
import json
import pandas as pd 
from typing import List
from sentence_transformers import SentenceTransformer

def split_text(text: str, max_chunk_length: int = 8000, overlap_ratio: float = 0.1) -> List[str]:
    if not (0 <= overlap_ratio < 1):
        raise ValueError("Overlap ratio must be between 0 and 1 (exclusive).")
    overlap_length = int(max_chunk_length * overlap_ratio)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_length, len(text))
        chunks.append(text[start:end])
        start += max_chunk_length - overlap_length
    return chunks

# embedding_model = SentenceTransformer("sentence-transformers/all-Mini-L6-v2")
embedding_model = SentenceTransformer("thenlper/gte-small")

class CSVEmbedder:
    """
    Handles CSV processing by reading data, computing embeddings,
    and storing them in a Chroma DB collection.
    """
    def __init__(self, collection_name: str, db_path: str, embedding_model, cache_size: int = 10_000_000_000):
        self.settings = Settings(
            chroma_segment_cache_policy="LRU",
            chroma_memory_limit_bytes=cache_size
        )
        self.client = chromadb.PersistentClient(path=db_path, settings=self.settings)
        self.collection = self.client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )
        if embedding_model is None:
            raise ValueError("An embedding model must be provided.")
        self.embedding_model = embedding_model
        self.id_counter = 0

    async def embed_csv(self, csv_file_path: str, batch_size: int = 100) -> None:
        """Embed CSV data in batches (wrapped in a thread for CPU work)."""
        await asyncio.to_thread(self._embed_csv_sync, csv_file_path, batch_size)

    def _embed_csv_sync(self, csv_file_path: str, batch_size: int):
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        df = pd.read_csv(csv_file_path)
        if 'id' not in df.columns:
            df['id'] = df.index.astype(str)
        rows = df.to_dict(orient='records')

        batch_ids = []
        batch_documents = []
        batch_metadatas = []

        for row in tqdm(rows, desc="Embedding CSV rows"):
            doc_id = str(row.get('id', self.id_counter))
            row_copy = {k: v for k, v in row.items() if k != 'id'}
            doc_text = json.dumps(row_copy)

            if len(doc_text) > 8000:
                for chunk in split_text(doc_text, max_chunk_length=8000, overlap_ratio=0.1):
                    batch_documents.append(chunk)
                    batch_ids.append(f"{doc_id}_{self.id_counter}")
                    batch_metadatas.append({"doc_name": os.path.basename(csv_file_path)})
                    self.id_counter += 1
            else:
                batch_documents.append(doc_text)
                batch_ids.append(doc_id)
                batch_metadatas.append({"doc_name": os.path.basename(csv_file_path)})
                self.id_counter += 1

            if len(batch_documents) >= batch_size:
                embeddings = [
                    self.embedding_model.encode(doc).tolist() for doc in batch_documents
                ]
                self.collection.add(
                    documents=batch_documents,
                    ids=batch_ids,
                    embeddings=embeddings,
                    metadatas=batch_metadatas
                )
                batch_ids, batch_documents, batch_metadatas = [], [], []

        if batch_documents:
            embeddings = [
                self.embedding_model.encode(doc).tolist() for doc in batch_documents
            ]
            self.collection.add(
                documents=batch_documents,
                ids=batch_ids,
                embeddings=embeddings,
                metadatas=batch_metadatas
            )
        print(f"[CSVEmbedder] Finished embedding CSV: {csv_file_path}")

    async def query_collection(self, query: str, n_results: int = 5) -> dict:
        """Query the Chroma collection for context using the query string."""
        query_embedding = self.embedding_model.encode(query).tolist()
        return await asyncio.to_thread(
            self.collection.query,
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

    async def delete_collection(self) -> None:
        """
        Deletes the entire collection from Chroma DB, effectively cleaning the embedded dataset.
        """
        # Wrapping the deletion call in asyncio.to_thread if it’s blocking.
        await asyncio.to_thread(self.client.delete_collection, self.collection.name)
        print(f"[CSVEmbedder] Collection '{self.collection.name}' deleted.")


