from typing import List, Tuple
from mcp.types import Prompt, PromptMessage
from anthropic.types import MessageParam

from core.chat import Chat
from core.claude import Claude
from mcp_client import MCPClient


class CliChat(Chat):
    def __init__(
        self,
        doc_client: MCPClient,
        clients: dict[str, MCPClient],
        claude_service: Claude,
    ):
        super().__init__(clients=clients, claude_service=claude_service)

        self.doc_client: MCPClient = doc_client

    async def list_prompts(self) -> list[Prompt]:
        return await self.doc_client.list_prompts()

    async def list_docs_ids(self) -> list[str]:
        """List available resource IDs - gracefully handles servers without docs://documents"""
        try:
            return await self.doc_client.read_resource("docs://documents")
        except Exception:
            # Weather server doesn't have docs://documents - return empty list
            # This allows CLI to work with any MCP server
            return []

    async def list_resource_uris(self) -> list[str]:
        """List all available MCP resource URIs for autocomplete.

        This uses the actual MCP list_resources() call to discover
        resources from any MCP server, not just document servers.
        """
        try:
            resources = await self.doc_client.list_resources()
            return [str(r.uri) for r in resources]
        except Exception:
            return []

    async def get_doc_content(self, doc_id: str) -> str:
        """Get document content - supports both doc server and weather server resources"""
        try:
            # Try document server format first
            return await self.doc_client.read_resource(f"docs://documents/{doc_id}")
        except Exception:
            # Try as direct URI (for weather server resources like file://readme)
            try:
                return await self.doc_client.read_resource(doc_id)
            except Exception:
                # Try with trailing slash (FastMCP adds / to file:// URIs)
                try:
                    return await self.doc_client.read_resource(f"{doc_id}/")
                except Exception:
                    return f"Could not read resource: {doc_id}"

    async def get_prompt(
        self, command: str, doc_id: str
    ) -> list[PromptMessage]:
        return await self.doc_client.get_prompt(command, {"doc_id": doc_id})

    async def _extract_resources(self, query: str) -> str:
        """Extract @mentioned resources from query"""
        mentions = [word[1:] for word in query.split() if word.startswith("@")]

        if not mentions:
            return ""

        doc_ids = await self.list_docs_ids()
        mentioned_docs: list[Tuple[str, str]] = []

        for mention in mentions:
            # Check if it's in the doc list (document server)
            if mention in doc_ids:
                content = await self.get_doc_content(mention)
                mentioned_docs.append((mention, content))
            else:
                # Try as direct URI (weather server: @file://readme)
                content = await self.get_doc_content(mention)
                if not content.startswith("Could not read"):
                    mentioned_docs.append((mention, content))

        return "".join(
            f'\n<document id="{doc_id}">\n{content}\n</document>\n'
            for doc_id, content in mentioned_docs
        )

    async def _process_command(self, query: str) -> bool:
        """Process /command style prompts - handles various argument patterns"""
        if not query.startswith("/"):
            return False

        words = query.split()
        command = words[0].replace("/", "")

        # Look up the prompt to check its arguments
        prompts = await self.list_prompts()
        prompt = next((p for p in prompts if p.name == command), None)

        if not prompt:
            print(f"Unknown prompt: /{command}")
            return False

        # Check for required arguments
        required_args = [arg for arg in (prompt.arguments or []) if arg.required]
        provided_args = words[1:]  # Arguments provided after the command

        if len(provided_args) < len(required_args):
            missing = [arg.name for arg in required_args[len(provided_args):]]
            print(f"Missing required argument(s) for /{command}: {', '.join(missing)}")
            print(f"Usage: /{command} " + " ".join(f"<{arg.name}>" for arg in required_args))
            return False

        # Build arguments dict by mapping positional args to prompt argument names
        args = {}
        if prompt.arguments:
            for i, arg in enumerate(prompt.arguments):
                if i < len(provided_args):
                    args[arg.name] = provided_args[i]

        try:
            messages = await self.doc_client.get_prompt(command, args)
            self.messages += convert_prompt_messages_to_message_params(messages)
            return True
        except Exception as e:
            print(f"Error executing prompt /{command}: {e}")
            return False

    async def _process_query(self, query: str):
        if await self._process_command(query):
            return

        added_resources = await self._extract_resources(query)

        prompt = f"""
        The user has a question:
        <query>
        {query}
        </query>

        The following context may be useful in answering their question:
        <context>
        {added_resources}
        </context>

        Note the user's query might contain references to documents like "@report.docx". The "@" is only
        included as a way of mentioning the doc. The actual name of the document would be "report.docx".
        If the document content is included in this prompt, you don't need to use an additional tool to read the document.
        Answer the user's question directly and concisely. Start with the exact information they need. 
        Don't refer to or mention the provided context in any way - just use it to inform your answer.
        """

        self.messages.append({"role": "user", "content": prompt})


def convert_prompt_message_to_message_param(
    prompt_message: "PromptMessage",
) -> MessageParam:
    role = "user" if prompt_message.role == "user" else "assistant"

    content = prompt_message.content

    # Check if content is a dict-like object with a "type" field
    if isinstance(content, dict) or hasattr(content, "__dict__"):
        content_type = (
            content.get("type", None)
            if isinstance(content, dict)
            else getattr(content, "type", None)
        )
        if content_type == "text":
            content_text = (
                content.get("text", "")
                if isinstance(content, dict)
                else getattr(content, "text", "")
            )
            return {"role": role, "content": content_text}

    if isinstance(content, list):
        text_blocks = []
        for item in content:
            # Check if item is a dict-like object with a "type" field
            if isinstance(item, dict) or hasattr(item, "__dict__"):
                item_type = (
                    item.get("type", None)
                    if isinstance(item, dict)
                    else getattr(item, "type", None)
                )
                if item_type == "text":
                    item_text = (
                        item.get("text", "")
                        if isinstance(item, dict)
                        else getattr(item, "text", "")
                    )
                    text_blocks.append({"type": "text", "text": item_text})

        if text_blocks:
            return {"role": role, "content": text_blocks}

    return {"role": role, "content": ""}


def convert_prompt_messages_to_message_params(
    prompt_messages: List[PromptMessage],
) -> List[MessageParam]:
    return [
        convert_prompt_message_to_message_param(msg) for msg in prompt_messages
    ]
