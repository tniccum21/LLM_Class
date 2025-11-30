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

Create `secrets.env` in the Week10 directory:

```env
ANTHROPIC_API_KEY=your-anthropic-key-here
CLAUDE_MODEL=claude-sonnet-4-20250514
SERPAPI_API_KEY=your-serpapi-key-here
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
| **Prompt** | `what_is_the_hip_weather_report` | Template - George Carlin weather persona |

### Section 4: Prompt + Resource Stacking

Prompts that reference other MCP primitives:

- `weather_with_context(location)` - Instructs LLM to check status, then get weather
- `multi_resource_prompt()` - Combines readme and status resources

### Section 5: Roots (Security)

File system boundaries for resource access:
- Configured in client MCP config (see [Roots Configuration](#roots-configuration) below)
- Limits where `file://` resources can read
- Security through principle of least privilege

### Section 6: Notifications & Logging

Server pushing updates to client:

- `get_weather_with_logging(location)` - Demonstrates `ctx.info()`, `ctx.warning()`, etc.
- Progress reporting via `ctx.report_progress()`

### Section 7: Sampling

Server requesting LLM from client:

- `get_weather_advice(location, activity)` - Uses `ctx.sample()` to get personalized advice
- Server provides data, client provides intelligence

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
await client.call_tool(name, args)      # Execute a tool
await client.read_resource(uri)          # Read resource data
await client.get_prompt(name, args)      # Get prompt template
```

## CLI Usage

### Basic Chat
```
> What's the weather like in Seattle?
```

### Using Prompts
```
> /what_is_the_hip_weather_report Portland
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

### "Module not found" errors
```bash
source ../.venv/bin/activate  # Ensure venv is active
```

### "SERPAPI_API_KEY not set"
```bash
# Check secrets.env exists and has the key
cat secrets.env
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

### MCP Config File Setup

Create `mcp_config.json` in the Week10 directory:

```json
{
  "mcpServers": {
    "weather": {
      "command": "python3",
      "args": ["stoopid_wx_server.py"],
      "roots": [
        "/absolute/path/to/Apps_with_AI/Week10",
        "/absolute/path/to/shared/data"
      ]
    }
  }
}
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
