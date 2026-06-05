# pipeline/embedder.py
import numpy as np
from sentence_transformers import SentenceTransformer

def load_embedding_model():
    """Cache in session state. Returns model once."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def embed_chunks(chunks: list, model) -> np.ndarray:
    """
    Stage 3: Convert all chunk texts to embeddings.
    Returns numpy array of shape (n_chunks, 384).
    """
    texts = [c["text"] for c in chunks]
    if not texts:
        return np.array([]).reshape(0, 384)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    return np.array(embeddings)
