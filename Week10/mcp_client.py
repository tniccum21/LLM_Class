import sys
import asyncio
from typing import Optional, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl
import json 

class MCPClient:
    def __init__(
        self,
        command: str,
        args: list[str],
        env: Optional[dict] = None,
    ):
        self._command = command
        self._args = args
        self._env = env
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
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(_stdio, _write)
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
        self, tool_name: str, tool_input: dict
    ) -> types.CallToolResult | None:
        # TODO: Call a particular tool and return the result
        return await self.session().call_tool(tool_name, tool_input)

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
