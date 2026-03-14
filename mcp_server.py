"""
MCP Server for Clinical Knowledge Base.
Provides tools for semantic search and full-text document retrieval.
"""

from mcp.server.fastmcp import FastMCP
from tools.knowledge_base_tool import search_knowledge_base, load_knowledge_base, get_collection_stats

# Create FastMCP server
mcp = FastMCP("Clinical-Knowledge-Base")

@mcp.tool()
def search_kb(query: str) -> str:
    """
    Search the internal clinical knowledge base (WHO guidelines, drug interactions).
    Returns relevant chunks with source attribution.
    """
    return search_knowledge_base(query)

@mcp.tool()
def get_full_docs() -> str:
    """
    Retrieve all clinical documents in full text. 
    Useful for comprehensive fact-checking.
    """
    return load_knowledge_base()

@mcp.resource("kb://stats")
def get_stats() -> str:
    """Get statistics about the knowledge base vector store."""
    stats = get_collection_stats()
    return f"Indexed Chunks: {stats['total_chunks']}\nFiles: {', '.join(stats['documents'])}"

if __name__ == "__main__":
    mcp.run()
