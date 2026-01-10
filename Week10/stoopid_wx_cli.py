import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp_client import MCPClient, create_roots_callback
from mcp import types
from mcp.shared.context import RequestContext
from core.claude import Claude

from core.cli_chat import CliChat
from core.cli import CliApp

# Load environment variables from secrets_client.env (client-specific secrets)
env_path = Path(__file__).parent / "secrets_client.env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # Fall back to .env

# Anthropic Config
claude_model = os.getenv("CLAUDE_MODEL", "")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")


assert claude_model, "Error: CLAUDE_MODEL cannot be empty. Update .env"
assert anthropic_api_key, (
    "Error: ANTHROPIC_API_KEY cannot be empty. Update .env"
)


# =============================================================================
# SECTION 7: SAMPLING HANDLER - Client provides LLM to server
# =============================================================================
#
# When a server calls ctx.sample(), it's asking the CLIENT to provide an
# LLM completion. This is the inverse of normal tool calls!
#
# Normal flow:  Client calls server's tools
# Sampling flow: Server requests client's LLM
#
# The sampling_handler receives:
#   - context: RequestContext with session info
#   - params: CreateMessageRequestParams containing:
#       - messages: List of SamplingMessage (role + content)
#       - systemPrompt: Optional system prompt
#       - maxTokens: Token limit requested
#       - temperature: Optional temperature setting
#       - modelPreferences: Optional model hints (client can ignore)
#
# The handler must return:
#   - CreateMessageResult with the LLM's response
#   - OR ErrorData if something went wrong
# =============================================================================

async def sampling_handler(
    context: RequestContext,
    params: types.CreateMessageRequestParams,
) -> types.CreateMessageResult | types.ErrorData:
    """
    Handle sampling requests from MCP servers.

    When a server calls ctx.sample(), this function is invoked.
    We use the Claude API to generate a response and return it
    to the server.

    This enables powerful patterns like:
    - Server fetches data, asks client LLM to analyze it
    - Server provides context, client provides intelligence
    - Separation of concerns: data layer vs reasoning layer
    """
    import anthropic

    print(f"\n🔄 [SAMPLING] Server requested LLM completion")
    print(f"   Messages: {len(params.messages)} message(s)")
    print(f"   Max tokens: {params.maxTokens}")
    if params.systemPrompt:
        print(f"   System prompt: {params.systemPrompt[:50]}...")

    try:
        # Initialize Anthropic client
        client = anthropic.Anthropic()

        # Convert MCP messages to Anthropic format
        # MCP uses: SamplingMessage with role and content
        # Anthropic uses: {"role": "user"|"assistant", "content": "..."}
        anthropic_messages = []
        for msg in params.messages:
            # Extract text from content (can be TextContent, ImageContent, etc.)
            if hasattr(msg.content, 'text'):
                content_text = msg.content.text
            elif isinstance(msg.content, str):
                content_text = msg.content
            else:
                content_text = str(msg.content)

            anthropic_messages.append({
                "role": msg.role,  # "user" or "assistant"
                "content": content_text
            })

        # Call Claude API
        # Note: We use our configured model, ignoring server's modelPreferences
        # The client controls which model is used - this is by design!
        response = client.messages.create(
            model=claude_model,
            max_tokens=params.maxTokens,
            system=params.systemPrompt or "",
            messages=anthropic_messages,
            temperature=params.temperature if params.temperature else 0.7,
        )

        # Extract response text
        response_text = response.content[0].text if response.content else ""

        print(f"   ✅ Generated response ({len(response_text)} chars)")

        # Return as CreateMessageResult
        # This gets sent back to the server's ctx.sample() call
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(
                type="text",
                text=response_text
            ),
            model=claude_model,
            stopReason="endTurn"
        )

    except Exception as e:
        print(f"   ❌ Sampling error: {e}")
        # Return error to server - it can handle gracefully
        return types.ErrorData(
            code=-1,
            message=f"Sampling failed: {str(e)}"
        )


async def main():
    claude_service = Claude(model=claude_model)

    server_scripts = sys.argv[1:]
    clients = {}

    # Point to the weather server instead of document server
    command, args = (
        ("uv", ["run", "stoopid_wx_server.py"])
        if os.getenv("USE_UV", "0") == "1"
        else ("python3", ["stoopid_wx_server.py"])
    )

    # ==========================================================================
    # ROOTS CONFIGURATION - Security boundaries for file:// resources
    # ==========================================================================
    # Load roots from MCP_ROOTS env var (comma-separated paths)
    # This defines which directories the server can access for file:// resources.
    #
    # The flow:
    # 1. Client loads roots from environment variable
    # 2. Client creates list_roots_callback from those paths
    # 3. Server calls ctx.list_roots() to get allowed directories
    # 4. Server enforces boundaries on file operations
    # ==========================================================================
    mcp_roots_env = os.getenv("MCP_ROOTS", "")
    roots = [r.strip() for r in mcp_roots_env.split(",") if r.strip()] if mcp_roots_env else []

    # If no roots configured, default to the Week10 directory for teaching
    if not roots:
        roots = [str(Path(__file__).parent.resolve())]
        print(f"ℹ️  No MCP_ROOTS in env, defaulting to: {roots[0]}")

    # Create the callback that will respond to server's roots/list request
    roots_callback = create_roots_callback(roots)

    async with AsyncExitStack() as stack:
        # Create MCP client WITH sampling_handler AND list_roots_callback
        # - sampling_callback: Enables server to request LLM completions
        # - list_roots_callback: Enables server to query allowed directories
        doc_client = await stack.enter_async_context(
            MCPClient(
                command=command,
                args=args,
                sampling_callback=sampling_handler,
                list_roots_callback=roots_callback,  # <-- Enable roots!
            )
        )
        clients["doc_client"] = doc_client

        for i, server_script in enumerate(server_scripts):
            client_id = f"client_{i}_{server_script}"
            client = await stack.enter_async_context(
                MCPClient(command="uv", args=["run", server_script])
            )
            clients[client_id] = client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            claude_service=claude_service,
        )

        cli = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
