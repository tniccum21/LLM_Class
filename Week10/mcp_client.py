import sys
import asyncio
from typing import Optional, Any, Callable, Awaitable
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from pydantic import AnyUrl, FileUrl
from pathlib import Path
import json

# Type alias for sampling callback function
# The server calls ctx.sample() -> client's sampling_handler is invoked
SamplingCallbackType = Callable[
    [RequestContext, types.CreateMessageRequestParams],
    Awaitable[types.CreateMessageResult | types.ErrorData]
]

# Type alias for logging callback function
# The server calls ctx.info(), ctx.debug(), ctx.warning(), ctx.error()
# -> client's logging_callback is invoked
LoggingCallbackType = Callable[
    [types.LoggingMessageNotificationParams],
    Awaitable[None]
]

# Type alias for list_roots callback function
# When the SERVER requests roots/list, the CLIENT responds with allowed directories
# This is a security feature - limits where file:// resources can read from
ListRootsCallbackType = Callable[
    [RequestContext],
    Awaitable[types.ListRootsResult | types.ErrorData]
]


def create_roots_callback(roots: list[str]) -> ListRootsCallbackType:
    """
    Create a list_roots_callback from a list of directory paths.

    This is the CLIENT-SIDE implementation of MCP roots:
    - Client DEFINES which directories the server can access
    - Server REQUESTS this list via roots/list
    - Server ENFORCES the boundaries when handling file:// resources

    Args:
        roots: List of absolute directory paths to allow access to

    Returns:
        A callback function that returns the roots as types.ListRootsResult

    Example:
        callback = create_roots_callback(["/path/to/Week10", "/path/to/data"])
        client = MCPClient(command="python3", args=["server.py"], list_roots_callback=callback)

    Security Note:
        - Only directories explicitly listed are accessible
        - Use absolute paths to avoid ambiguity
        - Follow principle of least privilege - grant only what's needed
    """
    # Convert paths to Root objects with file:// URIs
    root_objects = []
    for root_path in roots:
        # Resolve to absolute path
        abs_path = Path(root_path).resolve()
        # Create file:// URI
        file_uri = f"file://{abs_path}"
        root_objects.append(types.Root(
            uri=FileUrl(file_uri),
            name=abs_path.name  # Use directory name as human-readable name
        ))

    async def list_roots_callback(
        context: RequestContext,
    ) -> types.ListRootsResult | types.ErrorData:
        """Return the configured roots when server requests them."""
        return types.ListRootsResult(roots=root_objects)

    return list_roots_callback


def load_roots_from_env(env_var: str = "MCP_ROOTS") -> list[str]:
    """
    Load roots configuration from environment variable.

    Args:
        env_var: Name of environment variable containing comma-separated paths

    Returns:
        List of root directory paths

    Example in secrets_client.env:
        MCP_ROOTS=/path/to/Week10,/path/to/shared/data
    """
    import os
    roots_str = os.getenv(env_var, "")
    if not roots_str:
        return []
    return [r.strip() for r in roots_str.split(",") if r.strip()]


async def default_logging_callback(params: types.LoggingMessageNotificationParams) -> None:
    """
    Default handler for server logging notifications.

    When a server calls ctx.info(), ctx.debug(), ctx.warning(), or ctx.error(),
    this callback receives the notification and displays it to the user.

    The params contain:
    - level: "debug" | "info" | "notice" | "warning" | "error" | "critical" | "alert" | "emergency"
    - logger: Optional logger name
    - data: The actual log message (can be string, dict with 'msg' key, or other)
    """
    # Map log levels to display prefixes
    level_prefixes = {
        "debug": "🔍 [DEBUG]",
        "info": "ℹ️  [INFO]",
        "notice": "📝 [NOTICE]",
        "warning": "⚠️  [WARNING]",
        "error": "❌ [ERROR]",
        "critical": "🚨 [CRITICAL]",
        "alert": "🚨 [ALERT]",
        "emergency": "🚨 [EMERGENCY]",
    }

    prefix = level_prefixes.get(params.level, f"[{params.level.upper()}]")

    # Extract message from data
    # FastMCP sends data as {'msg': 'message', 'extra': None} format
    data = params.data
    if isinstance(data, dict) and 'msg' in data:
        message = data['msg']
    elif data is not None:
        message = str(data)
    else:
        message = ""

    print(f"{prefix} {message}")


class MCPClient:
    """
    MCP Client that connects to MCP servers and handles protocol operations.

    Supports:
    - sampling_callback: For servers that request LLM completions (ctx.sample())
    - logging_callback: For servers that send notifications (ctx.info(), ctx.debug(), etc.)
    - list_roots_callback: For servers that request allowed directories (roots/list)

    When logging_callback is not provided, uses default_logging_callback which
    prints server notifications to stdout with level-appropriate prefixes.

    Roots Security Model:
    - Client DEFINES allowed directories (roots)
    - Server REQUESTS roots via roots/list
    - Server ENFORCES access within those directories
    - This protects the file system from unauthorized access
    """
    def __init__(
        self,
        command: str,
        args: list[str],
        env: Optional[dict] = None,
        sampling_callback: Optional[SamplingCallbackType] = None,
        logging_callback: Optional[LoggingCallbackType] = None,
        list_roots_callback: Optional[ListRootsCallbackType] = None,
    ):
        self._command = command
        self._args = args
        self._env = env
        self._sampling_callback = sampling_callback
        # Use default logging callback if none provided - shows server notifications
        self._logging_callback = logging_callback or default_logging_callback
        # list_roots_callback: When provided, enables server to query allowed directories
        self._list_roots_callback = list_roots_callback
        self._session: Optional[ClientSession] = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()

    async def connect(self):
        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        _stdio, _write = stdio_transport

        # Pass callbacks to ClientSession:
        # - sampling_callback: Enables servers to request LLM completions (ctx.sample())
        # - logging_callback: Enables servers to send notifications (ctx.info(), etc.)
        # - list_roots_callback: Enables servers to query allowed directories (roots/list)
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(
                _stdio,
                _write,
                sampling_callback=self._sampling_callback,
                logging_callback=self._logging_callback,
                list_roots_callback=self._list_roots_callback,
            )
        )
        await self._session.initialize()

    def session(self) -> ClientSession:
        if self._session is None:
            raise ConnectionError(
                "Client session not initialized or cache not populated. Call connect_to_server first."
            )
        return self._session

    async def list_tools(self) -> list[types.Tool]:
        # TODO: Return a list of tools defined by the MCP server
        result = await self.session().list_tools()
        return result.tools
        
    async def call_tool(
        self, tool_name: str, tool_input: dict, progress_callback: Optional[Callable] = None
    ) -> types.CallToolResult | None:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            tool_input: Arguments to pass to the tool
            progress_callback: Optional callback for progress notifications.
                              Called with (progress: float, total: float | None, message: str | None)
                              when server calls ctx.report_progress()

        If no progress_callback is provided, uses a default that prints progress to stdout.
        """
        # Default progress callback - prints progress updates
        async def default_progress_callback(
            progress: float, total: float | None, message: str | None
        ) -> None:
            if total is not None:
                pct = int((progress / total) * 100)
                msg = f"📊 [PROGRESS] {pct}% ({progress}/{total})"
            else:
                msg = f"📊 [PROGRESS] {progress}"
            if message:
                msg += f" - {message}"
            print(msg)

        callback = progress_callback or default_progress_callback
        return await self.session().call_tool(tool_name, tool_input, progress_callback=callback)

    async def list_prompts(self) -> list[types.Prompt]:
        result = await self.session().list_prompts()
        return result.prompts


    async def get_prompt(self, prompt_name, args: dict[str, str]):
        result = await self.session().get_prompt(prompt_name, args)
        return result.messages

    async def list_resources(self) -> list[types.Resource]:
        """List all resources available from the MCP server."""
        result = await self.session().list_resources()
        return result.resources

    async def read_resource(self, uri: str) -> Any:
        # TODO: Read a resource, parse the contents and return it
        result = await self.session().read_resource(AnyUrl(uri))
        resource = result.contents[0]
        if isinstance(resource, types.TextResourceContents):
            if resource.mimeType == "application/json":
                return json.loads(resource.text)

            return resource.text


    async def cleanup(self):
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


# For testing - demonstrates all MCP client capabilities
async def main():
    async with MCPClient(
        # If using Python without UV, update command to 'python3' and remove "run" from args.
        command="python3",
        args=["stoopid_wx_server.py"],
    ) as _client:
        print("=" * 60)
        print("MCP Client Test - Discovering Server Capabilities")
        print("=" * 60)

        # Section 1: Tools
        print("\n📧 TOOLS (Actions the LLM can perform):")
        print("-" * 40)
        tools = await _client.list_tools()
        for t in tools:
            print(f"  • {t.name}")
            print(f"    {t.description[:60]}...")

        # Section 2: Resources
        print("\n📁 RESOURCES (Data the LLM can access):")
        print("-" * 40)
        resources = await _client.list_resources()
        for r in resources:
            print(f"  • {r.uri}")
            if r.name:
                print(f"    Name: {r.name}")

        # Section 3: Prompts
        print("\n💬 PROMPTS (Templates for LLM interactions):")
        print("-" * 40)
        prompts = await _client.list_prompts()
        for p in prompts:
            desc = p.description[:50] if p.description else 'No description'
            print(f"  • {p.name}: {desc}...")

        # Demo: Call a tool
        print("\n" + "=" * 60)
        print("🔧 DEMO: Calling get_weather tool")
        print("=" * 60)
        result = await _client.call_tool("get_weather", {"location": "Seattle"})
        # Extract just the text content for cleaner output
        if result and result.content:
            text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
            # Truncate long forecast data for readability
            if len(text) > 200:
                text = text[:200] + "..."
            print(f"  Result: {text}")

        # Demo: Read a resource
        print("\n" + "=" * 60)
        print("📖 DEMO: Reading data://app-status resource")
        print("=" * 60)
        status = await _client.read_resource("data://app-status")
        print(f"  Status: {status}")

        # Demo: Get a prompt
        print("\n" + "=" * 60)
        print("💬 DEMO: Getting weather_with_context prompt")
        print("=" * 60)
        prompt_msgs = await _client.get_prompt("weather_with_context", {"location": "Portland"})
        if prompt_msgs:
            content = prompt_msgs[0].content
            if hasattr(content, 'text'):
                print(f"  Prompt preview: {content.text[:150]}...")
            else:
                print(f"  Prompt preview: {str(content)[:150]}...")
        



if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
