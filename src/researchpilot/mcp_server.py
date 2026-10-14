
import json

from mcp.server.fastmcp import FastMCP

from researchpilot.corpus import get_document, search_documents


mcp = FastMCP("ResearchPilot Knowledge Tools")


@mcp.tool()
def search_documents_tool(query: str, top_k: int = 5) -> str:
    """
    Search the local research corpus.

    Returns a JSON string containing document IDs, titles,
    relevance scores, and text excerpts.
    """
    results = search_documents(query=query, top_k=top_k)
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
def get_document_tool(document_id: str) -> str:
    """
    Retrieve the complete text and metadata of a document by ID.
    """
    document = get_document(document_id)
    return json.dumps(document, ensure_ascii=False)


def main() -> None:
    # stdio is reserved for MCP protocol messages.
    # Do not print application logs to stdout in this process.
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
