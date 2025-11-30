# Week 10: MCP Clients - Teaching Outline for Slides

## Course Context
- **Previous Week**: Week 09 - Building MCP Servers with FastMCP
- **This Week**: Week 10 - Understanding MCP Clients
- **Duration**: ~45-60 minutes lecture + hands-on

---

## Slide Deck Structure

### Part 1: Introduction & Recap (5 min)

#### Slide 1: Title
- **Week 10: MCP Clients - The Other Side of the Protocol**
- Subtitle: "From Server Builder to Protocol Master"

#### Slide 2: Week 09 Recap
- We built an MCP **Server** (stoopid_wx_server.py)
- Server exposes: Tools, Resources, Prompts
- Question: How does the LLM actually USE these?

#### Slide 3: This Week's Goal
- Understand the **Client** perspective
- See both sides of the MCP conversation
- 7 key concepts to master

---

### Part 2: The MCP Client (10 min)

#### Slide 4: What is an MCP Client?
- The "other half" of the protocol
- Connects to servers via stdio/SSE/etc
- Discovers capabilities
- Executes operations

#### Slide 5: Architecture Diagram
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
- Show the bidirectional communication
- Highlight: Client initiates most requests

#### Slide 6: Client Operations Overview
| Operation | Purpose |
|-----------|---------|
| `list_tools()` | Discover available tools |
| `call_tool()` | Execute a tool |
| `list_resources()` | Discover data sources |
| `read_resource()` | Access resource data |
| `list_prompts()` | Discover prompt templates |
| `get_prompt()` | Retrieve a prompt |

#### Slide 7: Code Walkthrough - MCPClient Class
- Show `mcp_client.py` structure
- `connect()` - establishing the session
- `session()` - accessing the client session
- Async/await pattern

---

### Part 3: Tools - Client Perspective (5 min)

#### Slide 8: Tools Recap
- **Server side**: `@mcp.tool` decorator defines actions
- **Client side**: Discovers and calls tools

#### Slide 9: Tool Discovery
```python
tools = await client.list_tools()
for t in tools:
    print(f"{t.name}: {t.description}")
```
- Client receives tool metadata
- Name, description, parameters

#### Slide 10: Tool Execution
```python
result = await client.call_tool("get_weather", {"location": "Seattle"})
```
- Parameters sent as dict
- Returns `CallToolResult` with content

---

### Part 4: Resources - Client Perspective (5 min)

#### Slide 11: Resources Recap
- **Server side**: `@mcp.resource` defines data providers
- **Client side**: Discovers and reads resources

#### Slide 12: Resource Discovery
```python
resources = await client.list_resources()
for r in resources:
    print(f"{r.uri}: {r.name}")
```
- URIs identify resources: `file://readme/`, `data://app-status`, `file://config/`

#### Slide 13: Reading Resources
```python
content = await client.read_resource("data://app-status")
# Returns: {"status": "ok", "uptime": 12345}
```
- Content can be text or JSON
- MIME types indicate format

#### Slide 14: CLI Resource Autocomplete
- Type `@` in CLI to see available resources
- Autocomplete shows all MCP resources discovered via `list_resources()`
- Select resource to include in query context
```
> @file://readme/ tell me about this server
> @data://app-status what is the current status?
```

---

### Part 5: Prompts - Client Perspective (5 min)

#### Slide 15: Prompts Recap
- **Server side**: `@mcp.prompt` defines templates
- **Client side**: Retrieves and uses prompts

#### Slide 16: Prompt Discovery
```python
prompts = await client.list_prompts()
for p in prompts:
    print(f"{p.name}: {p.description}")
```

#### Slide 17: Getting Prompts
```python
messages = await client.get_prompt(
    "what_is_the_hip_weather_report",
    {"location": "Portland"}
)
```
- Returns `PromptMessage` list
- Ready to send to LLM

#### Slide 18: CLI Prompt Commands
- Type `/` to see available prompts
- Prompts are invoked as slash commands
```
> /what_is_the_hip_weather_report Portland
> /weather_with_context Seattle
> /multi_resource_prompt
```

---

### Part 6: Prompt + Resource Stacking (5 min)

#### Slide 19: Concept Introduction
- Prompts can REFERENCE other MCP primitives
- Create declarative workflows
- LLM orchestrates the execution

#### Slide 20: Stacking Pattern
```
Prompt → "Check status resource, then use weather tool"
         ↓
LLM reads data://app-status
         ↓
LLM calls get_weather("location")
         ↓
LLM synthesizes combined response
```

#### Slide 21: Code Example
```python
@mcp.prompt
def weather_with_context(location: str) -> str:
    return f"""
    1. FIRST: Check the 'data://app-status' resource
       - If status is not "ok", warn the user
    2. THEN: Get weather for {location} using get_weather tool
    3. FINALLY: Combine into comprehensive report
    """
```

#### Slide 22: Multi-Resource Stacking
```python
@mcp.prompt
def multi_resource_prompt() -> str:
    return """
    1. Read 'file://readme' for capabilities
    2. Read 'data://app-status' for current state
    3. Synthesize into system overview
    """
```

#### Slide 23: Why This Matters
- Reduces client-side complexity
- Server defines workflows declaratively
- LLM handles orchestration
- Reusable patterns across conversations

---

### Part 7: Roots - Security Boundaries (8 min)

#### Slide 24: The Security Problem
- Resources can read files: `file://readme`
- Without limits: Server could read ANY file
- Risk: `/etc/passwd`, `~/.ssh/id_rsa`, `~/.aws/credentials`

#### Slide 25: Roots Solution
- **Roots** = allowed directories for file resources
- Creates a "sandbox" for file operations
- Server declares needs, client grants access

#### Slide 26: Trust Model Diagram
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

#### Slide 27: MCP Config File Setup
Create `mcp_config.json`:
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

#### Slide 28: How file:// Resources Use Roots
```python
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

#### Slide 29: Claude Desktop Configuration
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

#### Slide 30: Security Best Practices
| Practice | Why |
|----------|-----|
| Use absolute paths | Avoid ambiguity |
| Principle of least privilege | Only grant what's needed |
| Never root system directories | `/etc`, `/usr`, `~/.ssh` are off-limits |
| Use project-relative roots | Keep access scoped to project |
| Review roots before granting | Understand what you're allowing |

#### Slide 31: Graceful Degradation
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

---

### Part 8: Notifications & Logging (5 min)

#### Slide 32: The Notification Pattern
- Normal: Client requests → Server responds
- Notifications: Server **pushes** to client
- Use case: Progress updates, warnings, debug info

#### Slide 33: Context Object
```python
@mcp.tool
async def my_tool(location: str, ctx: Context) -> str:
    await ctx.info("Starting process...")
    await ctx.report_progress(1, 3, "Step 1 complete")
    await ctx.warning("API rate limit approaching")
    return "Result"
```

#### Slide 34: Notification Methods
| Method | Purpose |
|--------|---------|
| `ctx.info()` | Informational message |
| `ctx.debug()` | Debug-level detail |
| `ctx.warning()` | Warning notification |
| `ctx.error()` | Error notification |
| `ctx.report_progress()` | Progress bar updates |

#### Slide 35: Client Handling
- Client decides how to display notifications
- Options: Log to console, update UI, ignore
- Fire-and-forget (no response needed)

---

### Part 9: Sampling - Server↔Client LLM (10 min)

#### Slide 36: The Big Idea
- Server can request LLM completions FROM the client
- Server has data, Client has LLM
- Best of both worlds!

> ✅ **Available**: `ctx.sample()` is available in **FastMCP 2.0+** (we have 2.13.1)

#### Slide 37: Sampling Workflow
```
1. Client calls server tool
2. Server fetches external data (SerpAPI weather)
3. Server calls ctx.sample("Analyze this data...")
4. Client's LLM generates response
5. Server receives LLM output
6. Server combines and returns final result
```

#### Slide 38: Code Example
```python
@mcp.tool
async def get_weather_advice(location: str, activity: str, ctx: Context) -> str:
    # Server's job: Get data
    weather = fetch_weather(location)

    # Client's job: Provide intelligence via ctx.sample()
    advice_response = await ctx.sample(
        messages=f"Weather: {weather}, Activity: {activity}. Provide advice...",
        max_tokens=200,
        temperature=0.7
    )

    # Extract text from response (TextContent | ImageContent | AudioContent)
    advice = advice_response.text

    # Combine both
    return f"Weather: {weather}\nAdvice: {advice}"
```

#### Slide 39: Why Sampling Matters
- Server doesn't need its own LLM or API keys
- Client controls which model is used
- Separation of concerns: Data vs Intelligence
- Security: Client can approve/deny sampling requests

#### Slide 40: Client Support & Fallbacks
| Client | Sampling Support |
|--------|------------------|
| Claude Desktop | ✅ Supported |
| Pydantic AI | ✅ Supported |
| Custom clients | Need `sampling_handler` |
| Our CLI (`stoopid_wx_cli.py`) | ⚠️ Falls back gracefully |

- Always wrap `ctx.sample()` in try/except for graceful fallback
- Tool remains functional even if client doesn't support sampling

---

### Part 10: Hands-On Demo (10 min)

#### Slide 41: Demo Setup
```bash
cd Week10
source ../.venv/bin/activate
python3 mcp_client.py
```

#### Slide 42: Demo 1 - Discovery
Show the client discovering:
- 3 tools: `get_weather`, `get_weather_with_logging`, `get_weather_advice`
- 3 resources: `file://readme/`, `data://app-status`, `file://config/`
- 4 prompts: `what_is_the_hip_weather_report`, `weather_with_context`, `multi_resource_prompt`

#### Slide 43: Demo 2 - Tool Execution
```python
result = await client.call_tool("get_weather", {"location": "Seattle"})
```
Show live weather data

#### Slide 44: Demo 3 - Full CLI
```bash
python3 stoopid_wx_cli.py
```
- Chat with the weather server
- Use `@` for resource autocomplete
- Use `/what_is_the_hip_weather_report Portland`
- Use `/weather_with_context Seattle`

#### Slide 45: Demo 4 - Resource Mentions
```
> @file://readme/ tell me about this server
> @data://app-status what is the current status?
```
Show how resources are injected into context

---

### Part 11: Summary & Key Takeaways (5 min)

#### Slide 46: The 7 Concepts
1. **Tools**: Client discovers and executes server functions
2. **Resources**: Client accesses server data by URI
3. **Prompts**: Client retrieves prompt templates
4. **Stacking**: Combine primitives for rich workflows
5. **Roots**: Security boundaries for file access
6. **Notifications**: Server pushes updates to client
7. **Sampling**: Server requests LLM from client via `ctx.sample()` (FastMCP 2.0+)

#### Slide 47: Mental Model
```
Client = Orchestrator (has LLM, coordinates)
Server = Specialist (has data, tools, templates)
MCP = Protocol (standard communication)
```

#### Slide 48: CLI Features Summary
| Feature | How to Use |
|---------|------------|
| Resource autocomplete | Type `@` to see available resources |
| Prompt commands | Type `/` to see available prompts |
| Resource mentions | `@file://readme/ <question>` |
| Prompt execution | `/weather_with_context Seattle` |

#### Slide 49: Next Steps
- Build your own MCP tools
- Connect multiple servers
- Implement sampling in real applications
- Configure roots for secure file:// resource access
- Explore MCP ecosystem (Claude Desktop, LM Studio, etc.)

---

## Hands-On Exercises

### Exercise 1: Basic Discovery (5 min)
Run `mcp_client.py` and identify all available tools, resources, and prompts.

### Exercise 2: Tool Execution (5 min)
Modify `mcp_client.py` to call `get_weather_with_logging` and observe the notifications.

### Exercise 3: Resource Autocomplete (5 min)
Run `stoopid_wx_cli.py` and:
- Type `@` to see available resources
- Select `@data://app-status` and ask about it
- Select `@file://readme/` and ask about it

### Exercise 4: Prompt Stacking (10 min)
Use the CLI to execute `/weather_with_context Seattle` and observe how the LLM combines resource and tool calls.

### Exercise 5: Add a New Resource (15 min)
Add a new resource to `stoopid_wx_server.py` that returns the current timestamp, then access it from the client using `@`.

### Exercise 6: Configure Roots (10 min)
Create a `mcp_config.json` file with roots configuration and understand how it limits file:// resource access.

---

## Additional Resources

- MCP Specification: https://spec.modelcontextprotocol.io/
- FastMCP Documentation: https://github.com/jlowin/fastmcp
- Anthropic MCP SDK: https://github.com/modelcontextprotocol/python-sdk

---

## Code Files for Reference

| File | Teaching Focus |
|------|----------------|
| `mcp_client.py` | Client implementation, discovery, execution |
| `stoopid_wx_server.py` | All 7 teaching sections with detailed docstrings |
| `stoopid_wx_cli.py` | Full application connecting client to server |
| `core/cli_chat.py` | Resource mentions, prompt commands, `list_resource_uris()` |
| `core/cli.py` | Autocomplete for `@` resources and `/` prompts |
| `mcp_config.json` | Roots configuration example |
| `README.md` | Full documentation including Roots Configuration |
