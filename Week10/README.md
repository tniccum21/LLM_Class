# Week 10: MCP Clients - The Other Side of the Protocol

This week we explore MCP from the **client perspective**. After Week 09's focus on building MCP servers, we now learn how clients connect to, discover, and interact with MCP servers.

## Learning Objectives

By the end of this week, you will understand:

1. **Tools** - How clients discover and call server-side functions
2. **Resources** - How clients access static data from servers
3. **Prompts** - How clients retrieve and use prompt templates
4. **Prompt + Resource Stacking** - Combining primitives for rich interactions
5. **Roots** - File system security boundaries for resources
6. **Notifications & Logging** - Server-to-client progress updates
7. **Sampling** - Server requesting LLM completions from client

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         MCP Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐         stdio          ┌──────────────────┐  │
│   │              │  ←──────────────────→  │                  │  │
│   │  MCP Client  │      MCP Protocol      │   MCP Server     │  │
│   │ (mcp_client) │                        │ (stoopid_wx)     │  │
│   │              │  list_tools()          │                  │  │
│   │              │  ←──────────────────   │  @mcp.tool       │  │
│   │              │  call_tool()           │                  │  │
│   │              │  ←──────────────────   │  @mcp.resource   │  │
│   │              │  list_resources()      │                  │  │
│   │              │  ←──────────────────   │  @mcp.prompt     │  │
│   │              │  read_resource()       │                  │  │
│   │              │  ←──────────────────   │                  │  │
│   │              │  get_prompt()          │                  │  │
│   └──────────────┘                        └──────────────────┘  │
│          │                                         │             │
│          │                                         │             │
│          ▼                                         ▼             │
│   ┌──────────────┐                        ┌──────────────────┐  │
│   │   CLI App    │                        │   External APIs  │  │
│   │  (cli_chat)  │                        │    (SerpAPI)     │  │
│   └──────────────┘                        └──────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.9+
- Anthropic API Key
- SerpAPI Key (for weather functionality)

## Quick Start

### 1. Configure Environment

Create two secrets files in the Week10 directory:

**`secrets_server.env`** (for the MCP server):
```env
SERPAPI_API_KEY=your-serpapi-key-here
```

**`secrets_client.env`** (for the CLI client):
```env
ANTHROPIC_API_KEY=your-anthropic-key-here
CLAUDE_MODEL=claude-sonnet-4-20250514

# Roots: comma-separated directories the server can access
MCP_ROOTS=/absolute/path/to/Apps_with_AI/Week10
```

### 2. Activate Virtual Environment

```bash
# From the Apps_with_AI root directory
source .venv/bin/activate
cd Week10
```

### 3. Test the MCP Client

```bash
python3 mcp_client.py
```

This demonstrates client-server communication:
- Discovers tools, resources, and prompts
- Calls the weather tool
- Reads a resource
- Gets a prompt template

### 4. Run the Full CLI

```bash
python3 stoopid_wx_cli.py
```

## Teaching Sections in Code

The `stoopid_wx_server.py` file is organized into 7 teaching sections:

### Section 1-3: Basic MCP Primitives

| Primitive | Example | Description |
|-----------|---------|-------------|
| **Tool** | `get_weather(location)` | Active function - fetches live weather data |
| **Resource** | `file://readme` | Passive data - returns documentation |
| **Resource** | `data://app-status` | Passive data - returns app status as JSON |
| **Prompt** | `hip_weather` | Template - George Carlin weather persona |

### Section 4: Prompt + Resource Stacking

Prompts that **fetch and embed** resource data at generation time:

| Prompt | Args | Description |
|--------|------|-------------|
| `weather_with_context` | `location` (required) | Embeds app status, instructs LLM to call weather tool |
| `multi_resource` | (none) | Embeds readme + status into system overview prompt |

#### Key Insight: Prompts EMBED Resources

The critical pattern here is that prompts **fetch resources server-side** and embed the data directly into the prompt text. The LLM receives pre-loaded context, not instructions to "read" resources.

```python
# Server-side: Prompt fetches and embeds resource
@mcp.prompt(name="weather_with_context", ...)
def weather_with_context_prompt(location: str) -> str:
    status_data = get_app_status()  # FETCH the resource
    return f"""
    APPLICATION STATUS: {json.dumps(status_data)}  # EMBED in prompt
    Now get weather for {location}...
    """
```

#### Using Prompts in the CLI

```bash
# Prompt with required argument
> /weather_with_context Seattle

# Prompt with no arguments
> /multi_resource
```

The `multi_resource` prompt demonstrates multi-resource embedding:
1. Fetches `file://readme` and embeds documentation
2. Fetches `data://app-status` and embeds status
3. LLM receives both pre-loaded for synthesis

### Section 5: Roots (Security)

File system boundaries for resource access:
- Configured in client MCP config (see [Roots Configuration](#roots-configuration) below)
- Limits where `file://` resources can read
- Security through principle of least privilege

### Section 6: Notifications & Logging

Server pushing updates to client:

- `get_weather_with_logging(location)` - Demonstrates `ctx.info()`, `ctx.warning()`, etc.
- Progress reporting via `ctx.report_progress()`
- `toggle_debug_mode` - Enable/disable debug notifications
- `get_debug_mode` - Check current debug state

#### Client-Side: Handling Notifications

The client needs callbacks to receive server notifications:

```python
# mcp_client.py - logging_callback for ctx.info(), ctx.debug(), etc.
async def default_logging_callback(params: types.LoggingMessageNotificationParams) -> None:
    level_prefixes = {
        "debug": "🔍 [DEBUG]",
        "info": "ℹ️  [INFO]",
        "warning": "⚠️  [WARNING]",
        "error": "❌ [ERROR]",
    }
    prefix = level_prefixes.get(params.level, f"[{params.level.upper()}]")

    # FastMCP sends data as {'msg': 'message'} format
    data = params.data
    message = data['msg'] if isinstance(data, dict) and 'msg' in data else str(data)
    print(f"{prefix} {message}")

# Pass to MCPClient - logging_callback is used by default
client = MCPClient(
    command="python3",
    args=["stoopid_wx_server.py"],
    logging_callback=default_logging_callback,  # Optional, uses default if omitted
)
```

#### Client-Side: Handling Progress

Progress updates from `ctx.report_progress()` are handled via `progress_callback`:

```python
# mcp_client.py - progress_callback for ctx.report_progress()
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

# Progress callback is passed to call_tool()
result = await client.call_tool(
    "get_weather_with_logging",
    {"location": "Seattle"},
    progress_callback=default_progress_callback  # Optional, uses default if omitted
)
```

### Section 7: Sampling

Server requesting LLM from client:

- `get_weather_advice(location, activity)` - Uses `ctx.sample()` to get personalized advice
- Server provides data, client provides intelligence

> **Note**: `ctx.sample()` is available in **FastMCP 2.0+**. The client must support sampling capability (Claude Desktop ✅, Pydantic AI ✅). If the client doesn't support sampling, the tool falls back gracefully.

#### Implementing a Sampling Handler

Our CLI implements `sampling_handler` in `stoopid_wx_cli.py` to support server-initiated LLM requests:

```python
async def sampling_handler(
    context: RequestContext,
    params: types.CreateMessageRequestParams,
) -> types.CreateMessageResult | types.ErrorData:
    """Handle sampling requests from MCP servers."""

    # Convert MCP messages to Anthropic format
    anthropic_messages = []
    for msg in params.messages:
        content_text = msg.content.text if hasattr(msg.content, 'text') else str(msg.content)
        anthropic_messages.append({"role": msg.role, "content": content_text})

    # Call Claude API
    response = client.messages.create(
        model=claude_model,
        max_tokens=params.maxTokens,
        system=params.systemPrompt or "",
        messages=anthropic_messages,
    )

    # Return as CreateMessageResult
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text=response.content[0].text),
        model=claude_model,
        stopReason="endTurn"
    )
```

Pass the handler when creating the client:

```python
doc_client = MCPClient(
    command="python3",
    args=["stoopid_wx_server.py"],
    sampling_callback=sampling_handler,  # Enable sampling!
)
```

The flow is:
1. Server tool calls `ctx.sample(messages=[...], max_tokens=200)`
2. MCP protocol sends sampling request to client
3. Client's `sampling_handler` receives the request
4. Handler calls Claude API and returns `CreateMessageResult`
5. Server receives the LLM response and continues

## Key Files

| File | Purpose |
|------|---------|
| `mcp_client.py` | Generic MCP client implementation |
| `stoopid_wx_server.py` | Teaching MCP server with 7 sections |
| `stoopid_wx_cli.py` | CLI that connects client to server |
| `core/cli_chat.py` | Chat logic with resource/prompt handling |
| `core/claude.py` | Anthropic Claude API wrapper |

## Client Methods Reference

```python
# Discovery
await client.list_tools()      # Get available tools
await client.list_resources()  # Get available resources
await client.list_prompts()    # Get available prompts

# Execution
await client.call_tool(name, args)                    # Execute a tool
await client.call_tool(name, args, progress_callback) # With progress handler
await client.read_resource(uri)                       # Read resource data
await client.get_prompt(name, args)                   # Get prompt template
```

## MCPClient Constructor

```python
MCPClient(
    command="python3",
    args=["server.py"],
    env=None,                    # Optional environment variables
    sampling_callback=None,      # Handler for ctx.sample() requests
    logging_callback=None,       # Handler for ctx.info()/debug()/warning()/error()
                                 # Uses default_logging_callback if not provided
)
```

## CLI Usage

### Basic Chat
```
> What's the weather like in Seattle?
```

### Using Prompts
```
> /hip_weather Portland
> /weather_with_context Seattle
> /multi_resource
```

### Resource Mentions
```
> @file://readme tell me about this server
```

## Development Notes

### Adding Custom Tools

In `stoopid_wx_server.py`:

```python
@mcp.tool(name="my_tool", description="...")
def my_tool(arg: str) -> str:
    return "result"
```

### Adding Resources

```python
@mcp.resource(uri="data://my-data")
def get_my_data() -> dict:
    return {"key": "value"}
```

### Adding Prompts

```python
@mcp.prompt
def my_prompt(location: str) -> str:
    return f"Instructions for {location}..."
```

## Troubleshooting

### "ctx Missing required argument" error

If you see this error when using `ctx: Context` in a tool:
```
ValidationError: 1 validation error for call[your_tool]
ctx
  Missing required argument
```

**Cause**: Wrong Context import. There are two `Context` classes with different module paths!

```python
# ❌ WRONG - this Context won't be auto-injected
from mcp.server.fastmcp import Context

# ✅ CORRECT - this Context gets auto-injected by FastMCP
from fastmcp import Context
```

FastMCP's `find_kwarg_by_type` looks for `fastmcp.server.context.Context`. Using the MCP wrapper breaks context injection.

### "Module not found" errors
```bash
source ../.venv/bin/activate  # Ensure venv is active
```

### "SERPAPI_API_KEY not set"
```bash
# Check secrets_server.env exists and has the key
cat secrets_server.env
```

### Python 2.7 errors
```bash
# Use python3 explicitly
python3 mcp_client.py
```

## Roots Configuration

Roots define file system boundaries for `file://` resources - a critical security feature that limits what directories an MCP server can access.

### The Security Problem

Without roots, a `file://` resource could potentially read ANY file on your system:
- `/etc/passwd` - system users
- `~/.ssh/id_rsa` - private keys
- `~/.aws/credentials` - cloud credentials

### The Solution: Roots

Roots create a sandbox - the server can ONLY access files within declared root directories.

### Roots Configuration

Configure roots in `secrets_client.env` using the `MCP_ROOTS` variable:

```env
# Single directory
MCP_ROOTS=/absolute/path/to/Apps_with_AI/Week10

# Multiple directories (comma-separated)
MCP_ROOTS=/path/to/Week10,/path/to/shared/data
```

### How file:// Resources Use Roots

Resources with `file://` URIs are restricted to the declared roots:

```python
# In stoopid_wx_server.py
@mcp.resource(uri="file://readme")
def get_readme() -> str:
    # This file MUST be within a declared root directory
    readme_path = Path(__file__).parent / "README.md"
    return readme_path.read_text()

@mcp.resource(uri="file://config")
def get_config_file() -> str:
    # Also restricted to root directories
    config_path = Path(__file__).parent / "config.json"
    return config_path.read_text() if config_path.exists() else "{}"
```

### Trust Model

```
┌─────────────────────────────────────────────────────────────┐
│                    ROOTS TRUST MODEL                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SERVER declares:     "I need access to /path/to/data"      │
│         ↓                                                   │
│  CLIENT decides:      "I'll grant /path/to/data"            │
│         ↓              OR "Denied - too broad"              │
│  ENFORCEMENT:         Client enforces the boundary          │
│                                                             │
│  Key Principle: Server requests, Client grants              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Best Practices

| Practice | Why |
|----------|-----|
| Use absolute paths | Avoid ambiguity |
| Principle of least privilege | Only grant what's needed |
| Never root system directories | `/etc`, `/usr`, `~/.ssh` are off-limits |
| Use project-relative roots | Keep access scoped to project |
| Review roots before granting | Understand what you're allowing |

### Example: Claude Desktop Configuration

For Claude Desktop, configure roots in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "stoopid-weather": {
      "command": "python3",
      "args": ["/path/to/Week10/stoopid_wx_server.py"],
      "roots": [
        "/path/to/Week10"
      ],
      "env": {
        "SERPAPI_API_KEY": "your-key-here"
      }
    }
  }
}
```

### Graceful Degradation

Resources should handle denied access gracefully:

```python
@mcp.resource(uri="file://config")
def get_config_file() -> str:
    config_path = Path(__file__).parent / "config.json"

    if config_path.exists():
        return config_path.read_text()
    else:
        # Return helpful fallback instead of crashing
        return '{"note": "No config.json found. Using defaults."}'
```

## Next Steps

After understanding MCP clients:
1. Build your own MCP tools
2. Combine multiple servers
3. Implement sampling in your applications
4. Add notifications for long-running operations
5. Configure roots for secure file:// resource access
