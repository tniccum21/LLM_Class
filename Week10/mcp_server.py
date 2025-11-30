from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from mcp.server.fastmcp.prompts import base


mcp = FastMCP("DocumentMCP", log_level="ERROR")


docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}

@mcp.tool(
    name="read_doc_contents",
    description="Reads the contents of a specified document by its ID.",
)
def read_document(
    doc_id: str = Field(description="The ID of the document to read."),
) -> str:
    if doc_id not in docs:
        raise ValueError(f"Document with ID {doc_id} not found.")

    return docs[doc_id]

# TODO: Write a tool to edit a doc
@mcp.tool(
    name="edit_doc",
    description="Edits the contents of a specified document by its ID.")
def edit_document(
    doc_id: str = Field(description="The ID of the document to edit."),
    old_string: str = Field(description="The string to be replaced in the document. Must match exactly including whitespace."),
    new_string: str = Field(description="The new text to replace the old text with.")
) -> None:
    if doc_id not in docs:
        raise ValueError(f"Document with ID {doc_id} not found.")

    docs[doc_id] = docs[doc_id].replace(old_string, new_string)
    
@mcp.resource(
    "docs://documents",
    mime_type="application/json",
)
def list_docs() -> dict[str]:
    return list(docs.keys())


@mcp.resource(
    "docs://documents/{doc_id}",
    mime_type="text/plain",
)
def get_doc(doc_id: str) -> str:
    if doc_id not in docs:
        raise ValueError(f"Document with ID {doc_id} not found.")
    return docs[doc_id]

# TODO: Write a prompt to rewrite a doc in markdown format
@mcp.prompt(
    name="format",
    description="Rewrite the provided document content in markdown format."
)
def format_doc(
    doc_id: str=Field(description="The ID of the document to format.")
) -> list[base.Message]:
    prompt = f"""
    Your goal is to reformat a document to be written with markdown syntax.

    The id of the document you need to reformat is 
    <document_id>
    {doc_id}
    </document_id>

    Add in headers, bullet points, tables, etc. where appropriate based on the content.
    Use the 'edit_document' tool to make changes to the document.
    """

    return [base.UserMessage(content=prompt)]

# TODO: Write a prompt to summarize a doc


if __name__ == "__main__":
    mcp.run(transport="stdio")
