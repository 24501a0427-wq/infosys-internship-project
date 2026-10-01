"""
MILESTONE 2 - Searchable Knowledge Base
---------------------------------------
Store document chunks as embeddings in ChromaDB
and retrieve relevant chunks using semantic search.
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "safety_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_embedder = None

def embed(texts):
    """Convert text into embedding vectors."""
    global _embedder
    if not texts:
        return []
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

def get_collection():
    """Get or create the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def add_document(chunks, source):
    """
    Add document chunks to the knowledge base.
    Existing chunks from the same source are removed first,
    preventing duplicate indexing.
    """
    chunks = [str(c).strip() for c in chunks if str(c).strip()]
    if not chunks:
        return 0
    col = get_collection()
    old = col.get(
        where={"source": source}
    ).get("ids", [])
    if old:
        col.delete(ids=old)
    col.add(
        ids=[f"{source}_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
        metadatas=[
            {"source": source, "chunk": i}
            for i in range(len(chunks))
        ],
    )
    return len(chunks)

def search(question, top_k=3):
    """Return the most relevant document chunks for a question."""
    if not question or not question.strip():
        return []
    col = get_collection()
    total = col.count()
    if total == 0:
        return []
    top_k = max(1, min(top_k, total))
    query_embedding = embed([question])[0]
    result = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    results = []
    for text, meta, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        results.append({
            "text": text,
            "source": meta["source"],
            "chunk": meta["chunk"],
            "distance": round(float(distance), 3),
        })
    return results

def list_documents():
    """Return {source filename: number of chunks}."""
    counts = {}
    for meta in get_collection().get()["metadatas"]:
        source = meta["source"]
        counts[source] = counts.get(source, 0) + 1
    return counts

def total_chunks():
    """Return the total number of stored chunks."""
    return get_collection().count()

if __name__ == "__main__":
    from documents import process_file
    manual = os.path.join(
        BASE_DIR,
        "data",
        "sample_safety_manual.txt"
    )
    chunks = process_file(manual)
    print("Adding:", len(chunks), "chunks")
    print("Stored:", add_document(chunks, "sample_safety_manual.txt"))
    print("\nSearch results:")
    for result in search("Do I need a helmet?", top_k=3):
        print(
            f"[{result['source']} #{result['chunk']}] "
            f"distance={result['distance']} "
            f"{result['text'][:100]}"
        )