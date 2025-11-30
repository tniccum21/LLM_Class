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
Client (AI App) ←→ MCP Protocol ←→ Server (Tools/Data)
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
- URIs identify resources: `file://readme`, `data://app-status`

#### Slide 13: Reading Resources
```python
content = await client.read_resource("data://app-status")
# Returns: {"status": "ok", "uptime": 12345}
```
- Content can be text or JSON
- MIME types indicate format

---

### Part 5: Prompts - Client Perspective (5 min)

#### Slide 14: Prompts Recap
- **Server side**: `@mcp.prompt` defines templates
- **Client side**: Retrieves and uses prompts

#### Slide 15: Prompt Discovery
```python
prompts = await client.list_prompts()
for p in prompts:
    print(f"{p.name}: {p.description}")
```

#### Slide 16: Getting Prompts
```python
messages = await client.get_prompt(
    "what_is_the_hip_weather_report",
    {"location": "Portland"}
)
```
- Returns `PromptMessage` list
- Ready to send to LLM

---

### Part 6: Prompt + Resource Stacking (5 min)

#### Slide 17: Concept Introduction
- Prompts can REFERENCE other MCP primitives
- Create declarative workflows
- LLM orchestrates the execution

#### Slide 18: Stacking Pattern
```
Prompt → "Check status resource, then use weather tool"
         ↓
LLM reads data://app-status
         ↓
LLM calls get_weather("location")
         ↓
LLM synthesizes combined response
```

#### Slide 19: Code Example
```python
@mcp.prompt
def weather_with_context(location: str) -> str:
    return f"""
    1. FIRST: Check the 'data://app-status' resource
    2. THEN: Get weather for {location} using get_weather tool
    3. FINALLY: Combine into comprehensive report
    """
```

#### Slide 20: Why This Matters
- Reduces client-side complexity
- Server defines workflows declaratively
- LLM handles orchestration
- Reusable patterns across conversations

---

### Part 7: Roots - Security Boundaries (5 min)

#### Slide 21: The Security Problem
- Resources can read files: `file://readme`
- Without limits: Server could read ANY file
- Risk: `/etc/passwd`, credentials, etc.

#### Slide 22: Roots Solution
- **Roots** = allowed directories for file resources
- Configured in client MCP settings
- Server declares needs, client grants access

#### Slide 23: Configuration Example
```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["server.py"],
      "roots": [
        "/path/to/Week10",
        "/path/to/shared/data"
      ]
    }
  }
}
```

#### Slide 24: Security Principles
- **Principle of Least Privilege**: Only grant what's needed
- **Trust Boundary**: Client decides what to allow
- **Graceful Degradation**: Resources should handle denied access

---

### Part 8: Notifications & Logging (5 min)

#### Slide 25: The Notification Pattern
- Normal: Client requests → Server responds
- Notifications: Server **pushes** to client
- Use case: Progress updates, warnings, debug info

#### Slide 26: Context Object
```python
@mcp.tool
async def my_tool(location: str, ctx: Context) -> str:
    await ctx.info("Starting process...")
    await ctx.report_progress(1, 3, "Step 1 complete")
    await ctx.warning("API rate limit approaching")
    return "Result"
```

#### Slide 27: Notification Methods
| Method | Purpose |
|--------|---------|
| `ctx.info()` | Informational message |
| `ctx.debug()` | Debug-level detail |
| `ctx.warning()` | Warning notification |
| `ctx.error()` | Error notification |
| `ctx.report_progress()` | Progress bar updates |

#### Slide 28: Client Handling
- Client decides how to display notifications
- Options: Log to console, update UI, ignore
- Fire-and-forget (no response needed)

---

### Part 9: Sampling - Server↔Client LLM (10 min)

#### Slide 29: The Big Idea
- Server can request LLM completions FROM the client
- Server has data, Client has LLM
- Best of both worlds!

#### Slide 30: Sampling Workflow
```
1. Client calls server tool
2. Server fetches external data
3. Server calls ctx.sample("Analyze this data...")
4. Client's LLM generates response
5. Server receives LLM output
6. Server combines and returns final result
```

#### Slide 31: Code Example
```python
@mcp.tool
async def get_weather_advice(location: str, activity: str, ctx: Context) -> str:
    # Server's job: Get data
    weather = fetch_weather(location)

    # Client's job: Provide intelligence
    advice = await ctx.sample(f"""
        Weather: {weather}
        Activity: {activity}
        Provide advice...
    """)

    # Combine both
    return f"Weather: {weather}\nAdvice: {advice}"
```

#### Slide 32: Why Sampling Matters
- Server doesn't need its own LLM
- Client controls which model is used
- Separation of concerns: Data vs Intelligence
- Security: Client can approve/deny sampling requests

#### Slide 33: Security Considerations
- Sampling requires client permission
- Client sees the sampling prompt
- Client can filter/modify requests
- Trust relationship between client and server

---

### Part 10: Hands-On Demo (10 min)

#### Slide 34: Demo Setup
```bash
cd Week10
source ../.venv/bin/activate
python3 mcp_client.py
```

#### Slide 35: Demo 1 - Discovery
Show the client discovering:
- 3 tools
- 3 resources
- 3 prompts

#### Slide 36: Demo 2 - Tool Execution
```python
result = await client.call_tool("get_weather", {"location": "Seattle"})
```
Show live weather data

#### Slide 37: Demo 3 - Full CLI
```bash
python3 stoopid_wx_cli.py
```
- Chat with the weather server
- Use `/what_is_the_hip_weather_report Portland`

---

### Part 11: Summary & Key Takeaways (5 min)

#### Slide 38: The 7 Concepts
1. **Tools**: Client discovers and executes server functions
2. **Resources**: Client accesses server data by URI
3. **Prompts**: Client retrieves prompt templates
4. **Stacking**: Combine primitives for rich workflows
5. **Roots**: Security boundaries for file access
6. **Notifications**: Server pushes updates to client
7. **Sampling**: Server requests LLM from client

#### Slide 39: Mental Model
```
Client = Orchestrator (has LLM, coordinates)
Server = Specialist (has data, tools, templates)
MCP = Protocol (standard communication)
```

#### Slide 40: Next Steps
- Build your own MCP tools
- Connect multiple servers
- Implement sampling in real applications
- Explore MCP ecosystem (Claude Desktop, LM Studio, etc.)

---

## Hands-On Exercises

### Exercise 1: Basic Discovery (5 min)
Run `mcp_client.py` and identify all available tools, resources, and prompts.

### Exercise 2: Tool Execution (5 min)
Modify `mcp_client.py` to call `get_weather_with_logging` and observe the notifications.

### Exercise 3: Prompt Stacking (10 min)
Use the CLI to execute `/weather_with_context Seattle` and observe how the LLM combines resource and tool calls.

### Exercise 4: Add a New Resource (15 min)
Add a new resource to `stoopid_wx_server.py` that returns the current timestamp, then access it from the client.

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
| `core/cli_chat.py` | Resource mentions, prompt commands |
