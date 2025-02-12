from typing import List

# ----------------------------
# Utility: Split long text into overlapping chunks
# ----------------------------
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
