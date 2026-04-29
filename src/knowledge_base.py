"""Knowledge base loader and retriever for RAG-enhanced recommendations."""

import json
import logging
import os
from typing import List, Dict, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")


def load_knowledge_base() -> List[Dict]:
    """Load all knowledge base documents from JSON files."""
    documents = []
    for filename in sorted(os.listdir(KB_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(KB_DIR, filename)
        with open(path, encoding="utf-8") as f:
            docs = json.load(f)
            documents.extend(docs)
    logger.info("Loaded %d knowledge base documents from %s", len(documents), KB_DIR)
    return documents


def get_embedding(text: str, client) -> List[float]:
    """Get an embedding vector from OpenAI for a single text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def build_kb_index(documents: List[Dict], client) -> Dict:
    """Build an embedding index for all knowledge base documents."""
    logger.info("Building embedding index for %d documents...", len(documents))
    texts = [f"{doc['title']}: {doc['content']}" for doc in documents]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    embeddings = np.array([item.embedding for item in response.data])
    logger.info("Embedding index built: shape %s", embeddings.shape)
    return {
        "documents": documents,
        "embeddings": embeddings,
        "texts": texts,
    }


def retrieve(query: str, index: Dict, client, top_k: int = 5) -> List[Dict]:
    """Retrieve the most relevant knowledge base documents for a query."""
    query_embedding = np.array(get_embedding(query, client)).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, index["embeddings"])[0]
    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for i in ranked_indices:
        doc = index["documents"][i]
        results.append({
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "category": doc["category"],
            "similarity": float(similarities[i]),
        })
    logger.info(
        "Retrieved %d documents for query '%s' (top score: %.3f)",
        len(results), query[:50], results[0]["similarity"] if results else 0,
    )
    return results
