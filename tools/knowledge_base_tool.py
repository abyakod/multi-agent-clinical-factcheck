"""
Knowledge Base Tool — MCP-Style Tool with ChromaDB Vector Store

Loads clinical documents, chunks them, and indexes into ChromaDB
for semantic search. This replaces the naive "load entire file" approach
with proper retrieval-augmented generation (RAG).

ChromaDB uses its built-in embedding model (all-MiniLM-L6-v2 by default).
"""

import os
from pathlib import Path
import chromadb

# Knowledge base directory
KB_DIR = Path(__file__).parent.parent / "knowledge_base"

# ChromaDB persistent storage
CHROMA_DIR = Path(__file__).parent.parent / ".chromadb"

# Initialize ChromaDB client
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = None

# MCP-style tool definition
TOOL_DEFINITION = {
    "name": "knowledge_base_search",
    "description": (
        "Search internal clinical knowledge base containing WHO Essential "
        "Medicines, drug interaction data, and clinical treatment guidelines."
    ),
    "parameters": {
        "query": {
            "type": "string",
            "description": "The clinical question to search for"
        }
    }
}


def _chunk_text(text: str, filename: str, chunk_size: int = 500) -> list:
    """Split text into chunks with metadata."""
    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "source": filename
            })
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "source": filename
        })

    return chunks


def _get_collection():
    """Get or create the ChromaDB collection, indexing docs if needed."""
    global _collection

    if _collection is not None:
        return _collection

    _collection = _client.get_or_create_collection(
        name="clinical_knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )

    # If collection is empty, index the knowledge base
    if _collection.count() == 0:
        print("📚 Indexing knowledge base into ChromaDB...")
        _index_documents()
        print(f"✅ Indexed {_collection.count()} chunks")

    return _collection


def _index_documents():
    """Load and index all knowledge base documents."""
    if not KB_DIR.exists():
        print("ERROR: Knowledge base directory not found.")
        return

    all_chunks = []
    for filepath in sorted(KB_DIR.glob("*.txt")):
        try:
            content = filepath.read_text(encoding="utf-8")
            chunks = _chunk_text(content, filepath.name)
            all_chunks.extend(chunks)
            print(f"   📄 {filepath.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"   ❌ Error loading {filepath.name}: {e}")

    if not all_chunks:
        return

    # Batch add to ChromaDB
    _collection.add(
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"]} for c in all_chunks]
    )


def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """
    Semantic search against the clinical knowledge base.

    Args:
        query: Clinical question to search for
        n_results: Number of results to return

    Returns:
        Formatted string of relevant passages with source attribution
    """
    collection = _get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    if not results["documents"][0]:
        return "No relevant information found in knowledge base."

    formatted = []
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]), 1
    ):
        source = meta.get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{doc}")

    return "\n\n---\n\n".join(formatted)


def load_knowledge_base() -> str:
    """
    Load ALL knowledge base documents (full text).
    Used by fact-checker which needs complete source docs.
    """
    documents = []
    if not KB_DIR.exists():
        return "ERROR: Knowledge base directory not found."

    for filepath in sorted(KB_DIR.glob("*.txt")):
        try:
            content = filepath.read_text(encoding="utf-8")
            documents.append(
                f"\n{'═' * 60}\n"
                f"DOCUMENT: {filepath.name}\n"
                f"{'═' * 60}\n"
                f"{content}"
            )
        except Exception as e:
            documents.append(f"\nERROR loading {filepath.name}: {e}\n")

    return "\n".join(documents) if documents else "No documents found."


def get_document_names() -> list:
    """Return available knowledge base document filenames."""
    if not KB_DIR.exists():
        return []
    return [f.name for f in sorted(KB_DIR.glob("*.txt"))]


def get_collection_stats() -> dict:
    """Return stats about the vector store."""
    collection = _get_collection()
    return {
        "total_chunks": collection.count(),
        "documents": get_document_names()
    }
