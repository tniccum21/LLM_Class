"""
Stoopid Weather Server - A Simple MCP Server Teaching Example
==============================================================

This module demonstrates how to build a Model Context Protocol (MCP) server that provides
weather information by integrating with Google Search via SerpAPI.

OVERVIEW:
---------
- Uses FastMCP framework for simplified MCP server creation
- Integrates with SerpAPI for Google Search weather queries
- Implements three MCP primitives:
  * Tool: get_weather() - Active function for weather queries
  * Resource: Static data providers (README, app status)
  * Prompt: Template generators for LLM interactions

PREREQUISITES:
--------------
- SERPAPI_API_KEY environment variable (stored in secrets_server.env)
- Required packages: fastmcp, serpapi, python-dotenv

"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from serpapi import Client

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
# FastMCP uses Python's logging module internally. By default it logs at DEBUG
# level, which shows "Sending INFO to client:" messages on stderr.
# Set to WARNING or higher to suppress these internal debug messages.
# This is separate from our ctx.debug() notifications to the client!

logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

# Load environment variables from secrets_server.env
# This keeps sensitive API keys out of version control
# Pattern: ALWAYS use .env type files for secrets, never hardcode credentials
env_path = Path(__file__).parent / "secrets_server.env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded environment variables from {env_path}")
else:
    print(f"⚠ Warning: {env_path} not found. Set SERPAPI_API_KEY manually.")

# ==============================================================================
# MCP SERVER INITIALIZATION
# ==============================================================================

# Create an MCP server instance
# FastMCP simplifies MCP server creation with decorators
# The server name appears in MCP client tools list
mcp = FastMCP("Stoopid Weather Server")

# ==============================================================================
# MCP TOOL IMPLEMENTATION
# ==============================================================================
# Tools are ACTIVE functions that LLMs can call to perform actions
# Think of tools as "verbs" - things the LLM can DO

@mcp.tool(
    name="get_weather",
    description=(
        "Get the current weather for a specified location. "
        "Use this tool when the user asks about weather, temperature, or climate conditions "
        "in any city or location. "
        "\n\nExamples:"
        "\n- 'What's the weather in Paris?'"
        "\n- 'Tell me the temperature in New York'"
        "\n- 'Is it raining in Seattle?'"
    )
)
def get_weather(location: str) -> str:
    """
    Fetch current weather information for a given location using Google Search.

    This function demonstrates:
    - API integration with error handling
    - Environment variable usage for credentials
    - Structured response parsing from search results
    - Graceful degradation when data is unavailable

    IMPLEMENTATION:
    --------------
    1. Validation Step: Do API credentials exist?
    2. Payload Construction: Build search query with parameters
    3. Execute API Call: Execute API call with error handling
    4. Parse Response: Extract relevant data
    5. Send Response to Host: Return human-readable result or error message

    Args:
        location (str): Name of city, region, or place to get weather for.
                       Examples: "Paris", "New York, NY", "Tokyo, Japan"

    Returns:
        str: Human-readable weather information string in format:
             "Weather in {location}: {temperature} and {conditions}"
             Or an error message if the request fails.

    Example Usage (from LLM):
        >>> get_weather("London")
        "Weather in London: 15°C and Partly Cloudy"

    Error Cases:
        - Missing API key: Returns credential error message
        - API error: Returns SerpAPI error details
        - No data found: Returns "No weather data found" message
        - Exception: Returns error with exception details

    Teaching Notes:
        - API keys should ALWAYS be in environment variables, never hardcoded
        - Always validate credentials before making API calls
        - Structure error handling from specific to general (API errors → exceptions)
        - Return user-friendly messages, not raw exceptions
    """
    # Step 1: Validate API credentials
    # Pattern: Fail fast if required credentials are missing
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY environment variable not set. Please set it in secrets_server.env"

    try:
        # Step 2: Initialize SerpAPI Client with API key
        # New serpapi (v0.1.5+) uses Client-based API
        client = Client(api_key=api_key)

        # Step 3: Construct search parameters for Google Search
        # - q: Search query string
        # - hl: Host language (en = English)
        # - gl: Geo-location (us = United States)
        params = {
            "q": f"What's the weather in {location}?",
            "hl": "en",
            "gl": "us"
        }

        # Step 4: Execute API call using new Client.search() method
        # engine="google" specifies Google Search
        results = client.search(engine="google", params=params)

        # Step 5: Handle API-level errors
        # SerpAPI returns {"error": "message"} for API failures
        if "error" in results:
            return f"Error from SerpAPI: {results['error']}"

        # Step 6: Parse response structure
        # Google Search results have an "answer_box" for direct answers
        # This contains structured weather data (temperature, conditions, etc.)
        answer_box = results.get("answer_box")
        if not answer_box:
            return f"No weather data found for {location}"

        # Step 7: Extract and format weather data
        # Use .get() with defaults to handle missing fields gracefully
        temperature = answer_box.get('temperature', 'N/A')
        conditions = answer_box.get('weather', 'N/A')
        forecast = answer_box.get('forecast', 'N/A')
        return f"Weather in {location}: {temperature} and {conditions} with forecast {forecast}"

    except Exception as e:
        # Step 8: Catch-all exception handler
        # Pattern: Log/return specific error details for debugging
        return f"Error getting weather for {location}: {str(e)}"


# ==============================================================================
# MCP RESOURCE IMPLEMENTATIONS
# ==============================================================================
# HELPER FUNCTIONS - Shared data access for Resources and Prompts
# ==============================================================================
# IMPORTANT: When prompts need to embed resource data, they can't call the
# @mcp.resource decorated function directly (it becomes a FunctionResource).
# Instead, create helper functions that both resources and prompts can use.

def _get_readme_data() -> str:
    """
    Helper function to get README content.

    This function is called by BOTH:
    - The @mcp.resource decorator (for client resource reads)
    - Prompts that need to EMBED this data

    Returns:
        str: README content or error message
    """
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text()
    else:
        return "README file not found."


def _get_app_status_data() -> dict:
    """
    Helper function to get application status data.

    This function is called by BOTH:
    - The @mcp.resource decorator (for client resource reads)
    - Prompts that need to EMBED this data

    Returns:
        dict: Application status information
    """
    return {
        "status": "ok",
        "uptime": 12345,  # In production, calculate actual uptime
        "hosted_at": mcp.settings.host
    }


# ==============================================================================
# Resources are PASSIVE data providers - static information the LLM can read
# Think of resources as "nouns" - things the LLM can ACCESS
# ==============================================================================

@mcp.resource(uri="file://readme")
def get_readme() -> str:
    """
    Provide the README documentation for this weather server.

    This resource demonstrates:
    - File-based resource loading
    - Path resolution relative to current file
    - Graceful handling of missing files

    Resources vs Tools:
    -------------------
    - Resources: Static data (documentation, config, datasets)
    - Tools: Active functions (API calls, computations, actions)

    MCP clients can list and read resources without executing code.

    Returns:
        str: Contents of README_stoopid_wx_server.md or error message.

    URI Scheme:
        file://readme - Custom URI identifying this resource
        Clients use this URI to request the resource content.

    Teaching Notes:
        - Resources are fetched by URI, not function name
        - Use Path for cross-platform file path handling
        - Always provide fallback for missing files
        - Resources are cacheable by MCP clients
    """
    return _get_readme_data()


@mcp.resource(
    uri="data://app-status",           # Required: Unique identifier for this resource
    name="ApplicationStatus",          # Optional: Human-readable name
    description="Provides the current status of the application."  # Optional: Description for clients
)
def get_application_status() -> dict:
    """
    Return application status and metadata as structured data.

    This resource demonstrates:
    - Explicit metadata specification (uri, name, description)
    - Returning structured data (dict) instead of plain text
    - Accessing MCP server settings (mcp.settings)

    Resource Metadata:
    ------------------
    - uri: Unique identifier (required) - data://app-status
    - name: Display name for clients - "ApplicationStatus"
    - description: Purpose explanation for LLM context

    Returns:
        dict: Status information with keys:
            - status (str): Application health ("ok", "degraded", "error")
            - uptime (int): Seconds since server start
            - hosted_at (str): Server host address

    Example Response:
        {
            "status": "ok",
            "uptime": 12345,
            "hosted_at": "localhost:8000"
        }

    Teaching Notes:
        - Resources can return any JSON-serializable data (dict, list, str, int)
        - Use structured data (dict) for complex information
        - mcp.settings provides server configuration access
        - Status endpoints are useful for health checks and monitoring
    """
    return _get_app_status_data()


# ==============================================================================
# MCP PROMPT IMPLEMENTATION
# ==============================================================================
# Prompts are TEMPLATE generators - reusable instruction patterns for LLMs
# Think of prompts as "prompt engineering as code"

@mcp.prompt(
    name="hip_weather",
    description="George Carlin 'Hippy-Dippy Weatherman' style report"
)
def hip_weather_prompt(location: str) -> str:
    """
    Generate a weather reporter prompt for the LLM
      in the style of George Carlin's 'Hippy-Dippy Weatherman from the 1960s'.

    This prompt demonstrates:
    - Parameterized prompt templates
    - Role-based prompting (weather reporter persona)
    - Prompt reusability across conversations

    Prompts vs Tools:
    -----------------
    - Tools: Execute code and return data
    - Prompts: Generate text instructions for the LLM

    Use prompts for:
    - Consistent role/persona application
    - Complex multi-step instructions
    - Reusable conversation starters

    Args:
        location (str): Location to generate weather report prompt for.

    Returns:
        str: Formatted prompt instructing LLM to act as weather reporter.

    Example Usage (from MCP client):
        Client lists available prompts → sees "hip_weather"
        Client calls prompt with location → receives formatted instruction
        Client sends instruction to LLM → LLM responds as 'hippy-dippy' weather reporter

    Notes:
        - Prompts are templates, not executables
        - MCP clients can discover and inject prompts into conversations
        - Use prompts for consistent behavior across sessions
        - Combine prompts with tools for powerful workflows:
          1. Use prompt to set role
          2. Use tool to fetch data
          3. LLM synthesizes response in character
    """
    return f"""You are a weather reporter. Give the weather report for {location} in the style of George Carlin's 'Hippy-Dippy Weatherman from the 1960s'?"""


# ==============================================================================
# SECTION 4: PROMPT + RESOURCE STACKING
# ==============================================================================
# Combining prompts with resources creates powerful context-aware interactions
# This demonstrates how MCP primitives work TOGETHER, not in isolation

@mcp.prompt(
    name="weather_with_context",
    description="Check app status, then fetch weather for location"
)
def weather_with_context_prompt(location: str) -> str:
    """
    Generate a weather prompt that EMBEDS resource data directly.

    This demonstrates PROMPT + RESOURCE STACKING:
    - The prompt INCLUDES some resource data (not just references it)
    - This is the KEY insight: prompts can fetch and embed resources!

    STACKING PATTERN:
    -----------------
    Prompt fetches Resource → Embeds data → Instructs LLM to use Tool

    Workflow when this prompt is used:
    1. Client requests this prompt with location
    2. SERVER fetches data://app-status resource
    3. Prompt returns with embedded status + instructions
    4. LLM sees status context AND gets instruction to call weather tool
    5. LLM synthesizes both into cohesive response

    Args:
        location: The location for the weather report

    Returns:
        str: A prompt with embedded resource data and tool instructions

    Teaching Notes:
        - Prompts can FETCH and EMBED resource data at generation time
        - The LLM receives pre-fetched context, not just instructions
        - This is more reliable than telling LLM to "read" resources
        - Stacking = Server does the orchestration, not the LLM
    """
    # FETCH the resource data and embed it in the prompt
    status_data = _get_app_status_data()

    return f"""
    You are providing a weather report with application context.

    APPLICATION STATUS (from data://app-status resource):
    {json.dumps(status_data, indent=2)}

    INSTRUCTIONS:
    1. Review the application status above
       - If status is not "ok", warn the user about potential data quality issues
       - Note the uptime and server information

    2. Get the current weather for {location} using the get_weather tool

    3. Provide a comprehensive weather report that includes:
       - The current conditions from the tool
       - A note about data freshness based on the app status
       - A friendly recommendation for the day

    Combine all information into a cohesive, helpful response.
    """


@mcp.prompt(
    name="multi_resource",
    description="Synthesize readme + status into system overview"
)
def multi_resource_prompt() -> str:
    """
    A prompt demonstrating MULTI-RESOURCE EMBEDDING.

    This shows advanced stacking where the prompt:
    - FETCHES the README resource for context about capabilities
    - FETCHES the app-status resource for current state
    - EMBEDS both into the prompt for the LLM

    Use Case: System documentation and status queries

    Returns:
        str: Prompt with embedded multi-resource data

    Teaching Notes:
        - Prompts can FETCH and EMBED multiple resources
        - Server does the orchestration, LLM gets pre-loaded context
        - More reliable than instructing LLM to read resources
    """
    # FETCH both resources and embed them
    readme_content = _get_readme_data()
    status_data = _get_app_status_data()

    return f"""
    You are providing a comprehensive system overview.

    === SERVICE DOCUMENTATION (from file://readme resource) ===
    {readme_content[:2000]}  # Truncate for prompt size
    {"... (truncated)" if len(readme_content) > 2000 else ""}

    === APPLICATION STATUS (from data://app-status resource) ===
    {json.dumps(status_data, indent=2)}

    INSTRUCTIONS:
    Based on the documentation and status above, provide a professional
    system status report that helps a user understand:
    1. WHAT this service does and its capabilities
    2. WHETHER it's currently operational
    3. Key features and how to use them

    Be concise but comprehensive.
    """


# ==============================================================================
# SECTION 5: ROOTS (Resource File System Boundaries)
# ==============================================================================
# Roots define the file system boundaries that resources can access
# They're a SECURITY feature - limiting where file:// resources can read from

# NOTE: Roots are configured at SERVER STARTUP, not via decorators
# This section documents how roots work conceptually

"""
ROOTS DOCUMENTATION
===================

What are Roots?
---------------
Roots define the directories that file:// resources are allowed to access.
Think of them as a "sandbox" for file operations.

Without roots: file://readme could potentially read ANY file on the system
With roots: file://readme can ONLY read files within declared root directories

Configuration Example (in MCP client config):
---------------------------------------------
{
    "mcpServers": {
        "weather": {
            "command": "python",
            "args": ["stoopid_wx_server.py"],
            "roots": [
                "/path/to/Week10",           # Allow access to Week10 directory
                "/path/to/shared/data"       # Allow access to shared data
            ]
        }
    }
}

FastMCP Root Access:
--------------------
In FastMCP, you can access declared roots via:
    mcp.settings.roots  # List of root directories

Security Implications:
----------------------
- ALWAYS use roots in production
- Never allow root access to system directories (/etc, /usr, etc.)
- Principle of least privilege: only declare roots you actually need
- Roots are enforced by the MCP CLIENT, not the server
  (Server declares what it needs, client decides what to grant)

Teaching Notes:
---------------
- Roots are about TRUST between client and server
- Client can reject root requests that seem too broad
- Use relative paths within roots for portability
- Resources should gracefully handle denied root access
"""

# ==============================================================================
# ROOTS IMPLEMENTATION: Request roots from client and enforce boundaries
# ==============================================================================

# Server-side cache for roots (populated when tools need them)
_cached_roots: list[str] | None = None


def is_path_within_roots(file_path: Path, roots: list[str]) -> bool:
    """
    Check if a file path is within any of the allowed root directories.

    This is the SERVER-SIDE ENFORCEMENT of the roots security model:
    - Client DEFINES which directories are allowed (via list_roots_callback)
    - Server REQUESTS those roots and VALIDATES all file access against them

    Args:
        file_path: The absolute path to check
        roots: List of allowed root directory paths

    Returns:
        bool: True if the path is within at least one root, False otherwise

    Security Note:
        - Always resolve paths to absolute to prevent directory traversal attacks
        - Check that the file path STARTS WITH a root path (is a subdirectory)
    """
    file_path = file_path.resolve()
    for root in roots:
        root_path = Path(root).resolve()
        # Check if file_path is equal to or under root_path
        try:
            file_path.relative_to(root_path)
            return True
        except ValueError:
            continue
    return False


@mcp.tool(
    name="get_roots",
    description="Query the client for allowed root directories (MCP roots/list)"
)
async def get_roots_tool(ctx: Context) -> str:
    """
    Demonstrate how a server requests roots from the client.

    This is the KEY TEACHING POINT for roots:
    - Server calls ctx.list_roots() to ask the client "What directories can I access?"
    - Client responds with a list of Root objects (uri + optional name)
    - Server then ENFORCES these boundaries on all file:// operations

    The MCP Protocol Flow:
        Server: "roots/list" request → Client
        Client: ListRootsResult (list of Root objects) → Server

    Returns:
        str: JSON showing the roots configuration

    Teaching Notes:
        - If client doesn't provide roots, the list will be empty
        - Empty roots = server has no explicit file access permissions
        - Production servers should ALWAYS check roots before file operations
    """
    global _cached_roots

    try:
        roots = await ctx.list_roots()

        # Cache the roots for use by other tools/resources
        _cached_roots = [str(r.uri).replace("file://", "") for r in roots]

        if not roots:
            return json.dumps({
                "status": "warning",
                "message": "No roots configured by client",
                "roots": [],
                "implication": "Server has no explicit file access permissions"
            }, indent=2)

        root_info = []
        for r in roots:
            root_info.append({
                "uri": str(r.uri),
                "name": r.name,
                "path": str(r.uri).replace("file://", "")
            })

        return json.dumps({
            "status": "ok",
            "message": f"Client granted access to {len(roots)} root(s)",
            "roots": root_info
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to get roots: {str(e)}",
            "note": "Client may not support roots capability"
        }, indent=2)


@mcp.tool(
    name="read_file_safe",
    description="Read a file with roots enforcement (demonstrates secure file access)"
)
async def read_file_safe(file_path: str, ctx: Context) -> str:
    """
    Demonstrate SECURE file reading with roots enforcement.

    This tool shows the proper pattern for file access:
    1. Request roots from client (if not cached)
    2. Validate requested path is within allowed roots
    3. Only then read the file

    Args:
        file_path: Path to the file to read (absolute or relative to server)
        ctx: MCP Context for requesting roots

    Returns:
        str: File contents if allowed, error message if denied

    Security Pattern:
        if is_path_within_roots(path, roots):
            return read_file(path)
        else:
            raise AccessDenied("Path not within allowed roots")

    Teaching Notes:
        - This is how production MCP servers should handle file access
        - Always validate BEFORE reading, not after
        - Provide clear error messages for debugging
    """
    global _cached_roots

    # Get roots from client if not cached
    if _cached_roots is None:
        try:
            roots = await ctx.list_roots()
            _cached_roots = [str(r.uri).replace("file://", "") for r in roots]
        except Exception:
            _cached_roots = []

    # Resolve the requested path
    requested_path = Path(file_path)
    if not requested_path.is_absolute():
        # Treat relative paths as relative to server directory
        requested_path = Path(__file__).parent / file_path
    requested_path = requested_path.resolve()

    # SECURITY CHECK: Validate path is within roots
    if not _cached_roots:
        await ctx.warning(f"No roots configured - file access denied for security")
        return json.dumps({
            "status": "denied",
            "reason": "No roots configured by client",
            "requested_path": str(requested_path),
            "suggestion": "Configure MCP_ROOTS in secrets_client.env"
        }, indent=2)

    if not is_path_within_roots(requested_path, _cached_roots):
        await ctx.warning(f"Access denied: {requested_path} not in allowed roots")
        return json.dumps({
            "status": "denied",
            "reason": "Path not within allowed roots",
            "requested_path": str(requested_path),
            "allowed_roots": _cached_roots
        }, indent=2)

    # Path is valid - read the file
    await ctx.info(f"Access granted: {requested_path}")
    if not requested_path.exists():
        return json.dumps({
            "status": "not_found",
            "path": str(requested_path)
        }, indent=2)

    try:
        content = requested_path.read_text()
        return json.dumps({
            "status": "ok",
            "path": str(requested_path),
            "size": len(content),
            "content": content[:1000] + ("..." if len(content) > 1000 else "")
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "path": str(requested_path),
            "error": str(e)
        }, indent=2)


@mcp.tool(
    name="list_files",
    description="List files in a directory with roots enforcement (demonstrates secure directory listing)"
)
async def list_files_tool(directory: str = "", ctx: Context = None) -> str:
    """
    List files and directories within allowed roots.

    This tool demonstrates the standard MCP pattern for directory listing:
    - MCP doesn't have a built-in "list directory" primitive
    - Instead, implement as a TOOL that respects roots boundaries
    - Same security pattern as read_file_safe

    Args:
        directory: Directory path to list (absolute, relative to server, or empty for roots)
                   If empty, lists contents of each allowed root
        ctx: MCP Context for requesting roots

    Returns:
        str: JSON with directory contents or error message

    Teaching Notes:
        - Directory listing is a common need but NOT a built-in MCP primitive
        - Implement as a tool that enforces roots security
        - Empty directory = show what's available in the allowed roots
        - This enables file discovery within the security sandbox
    """
    global _cached_roots

    # Get roots from client if not cached
    if _cached_roots is None:
        try:
            roots = await ctx.list_roots()
            _cached_roots = [str(r.uri).replace("file://", "") for r in roots]
        except Exception:
            _cached_roots = []

    # No roots = no access
    if not _cached_roots:
        await ctx.warning("No roots configured - directory access denied")
        return json.dumps({
            "status": "denied",
            "reason": "No roots configured by client",
            "suggestion": "Configure MCP_ROOTS in secrets_client.env"
        }, indent=2)

    # If no directory specified, list contents of all roots
    if not directory:
        await ctx.info("Listing contents of all allowed roots")
        result = {
            "status": "ok",
            "roots": []
        }
        for root in _cached_roots:
            root_path = Path(root)
            if root_path.exists() and root_path.is_dir():
                entries = []
                try:
                    for entry in sorted(root_path.iterdir()):
                        entries.append({
                            "name": entry.name,
                            "type": "directory" if entry.is_dir() else "file",
                            "size": entry.stat().st_size if entry.is_file() else None
                        })
                except PermissionError:
                    entries = [{"error": "Permission denied"}]

                result["roots"].append({
                    "path": str(root_path),
                    "entries": entries
                })
            else:
                result["roots"].append({
                    "path": str(root_path),
                    "error": "Not found or not a directory"
                })
        return json.dumps(result, indent=2)

    # Resolve the requested directory path
    requested_path = Path(directory)
    if not requested_path.is_absolute():
        # Treat relative paths as relative to server directory
        requested_path = Path(__file__).parent / directory
    requested_path = requested_path.resolve()

    # SECURITY CHECK: Validate path is within roots
    if not is_path_within_roots(requested_path, _cached_roots):
        await ctx.warning(f"Access denied: {requested_path} not in allowed roots")
        return json.dumps({
            "status": "denied",
            "reason": "Path not within allowed roots",
            "requested_path": str(requested_path),
            "allowed_roots": _cached_roots
        }, indent=2)

    # Path is valid - list directory
    await ctx.info(f"Listing directory: {requested_path}")

    if not requested_path.exists():
        return json.dumps({
            "status": "not_found",
            "path": str(requested_path)
        }, indent=2)

    if not requested_path.is_dir():
        return json.dumps({
            "status": "error",
            "path": str(requested_path),
            "error": "Not a directory"
        }, indent=2)

    try:
        entries = []
        for entry in sorted(requested_path.iterdir()):
            entry_info = {
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
            }
            if entry.is_file():
                entry_info["size"] = entry.stat().st_size
            entries.append(entry_info)

        return json.dumps({
            "status": "ok",
            "path": str(requested_path),
            "count": len(entries),
            "entries": entries
        }, indent=2)
    except PermissionError:
        return json.dumps({
            "status": "error",
            "path": str(requested_path),
            "error": "Permission denied"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "path": str(requested_path),
            "error": str(e)
        }, indent=2)


# Example resource that demonstrates roots - now with enforcement
@mcp.resource(uri="file://config")
def get_config_file() -> str:
    """
    Demonstrate a resource that reads from the file system.

    NOTE: Resources run synchronously and can't call ctx.list_roots().
    For resources, roots enforcement would typically be done at the
    MCP transport layer or by using tools instead.

    In production with roots configured:
    - Use tools (like read_file_safe) for secure file access
    - Or implement roots checking at the resource registration level

    Returns:
        str: Config file contents or error message
    """
    config_path = Path(__file__).parent / "config.json"

    if config_path.exists():
        return config_path.read_text()
    else:
        # Return helpful message - config is optional
        return '{"note": "No config.json found. Using defaults."}'


# ==============================================================================
# SECTION 6: SERVER→CLIENT NOTIFICATIONS & LOGGING
# ==============================================================================
# Notifications allow the SERVER to push information to the CLIENT
# This is the reverse of normal request/response - server initiates!

# FastMCP provides built-in logging that becomes MCP notifications
# IMPORTANT: Context is imported at the top from fastmcp, not mcp.server.fastmcp
# These are different types and using the wrong one breaks ctx injection!

# ==============================================================================
# SERVER STATE: Debug Mode Toggle
# ==============================================================================
# This demonstrates how a server can maintain state that affects its behavior.
# In a real server, you might use this for:
# - Verbose/quiet modes
# - Feature flags
# - User preferences
#
# Note: This state is per-server-instance. If the server restarts, it resets.
# ==============================================================================

_debug_mode_enabled = True  # Default: debug messages are shown


@mcp.tool(
    name="toggle_debug_mode",
    description="Toggle debug message visibility on/off. Returns current state after toggle."
)
async def toggle_debug_mode(ctx: Context) -> str:
    """
    Toggle whether debug-level notifications are shown.

    SERVER STATE CONCEPT:
    ---------------------
    MCP servers can maintain internal state that persists across tool calls.
    This tool demonstrates:
    1. Server-side state management (the _debug_mode_enabled flag)
    2. Tools that modify server behavior
    3. Using notifications to confirm state changes

    When debug mode is OFF:
    - ctx.debug() calls are suppressed (not sent to client)
    - ctx.info(), ctx.warning(), ctx.error() still work normally
    - ctx.report_progress() still works normally

    When debug mode is ON (default):
    - All notification levels are sent to client

    Args:
        ctx: MCP Context for sending notifications

    Returns:
        str: Current debug mode state after toggling

    Teaching Notes:
        - Server state persists for the lifetime of the server process
        - Tools can read AND modify server state
        - This is useful for: feature flags, user preferences, caching
        - State is NOT shared between server instances
    """
    global _debug_mode_enabled

    # Toggle the state
    _debug_mode_enabled = not _debug_mode_enabled

    # Notify about the change
    new_state = "ON" if _debug_mode_enabled else "OFF"
    await ctx.info(f"🔧 Debug mode toggled: now {new_state}")

    if _debug_mode_enabled:
        await ctx.debug("🔍 Debug messages like this one will now appear")
    else:
        await ctx.info("ℹ️  Debug messages are now hidden (this is an INFO message)")

    return f"Debug mode is now {new_state}. Debug-level notifications are {'visible' if _debug_mode_enabled else 'hidden'}."


@mcp.tool(
    name="get_debug_mode",
    description="Check current debug mode state without changing it."
)
async def get_debug_mode() -> str:
    """
    Get the current debug mode state.

    This is a simple read-only tool that returns server state.
    Useful for checking configuration without side effects.

    Returns:
        str: Current debug mode state
    """
    state = "ON" if _debug_mode_enabled else "OFF"
    return f"Debug mode is currently {state}"


async def debug_log(ctx: Context, message: str) -> None:
    """
    Helper function that respects the debug mode toggle.

    Use this instead of ctx.debug() directly when you want debug messages
    to be controllable via toggle_debug_mode.

    Args:
        ctx: MCP Context
        message: Debug message to send (only if debug mode is ON)
    """
    if _debug_mode_enabled:
        await ctx.debug(message)


@mcp.tool(
    name="get_weather_with_logging",
    description="Get weather with detailed progress notifications. Use this to see server-side logging in action."
)
async def get_weather_with_logging(location: str, ctx: Context) -> str:
    """
    Weather tool that demonstrates SERVER→CLIENT notifications.

    NOTIFICATIONS CONCEPT:
    ----------------------
    Normal flow: Client requests → Server responds
    With notifications: Server can ALSO send progress updates, logs, warnings

    The Context object (ctx) provides:
    - ctx.info(msg)    → Informational notification
    - ctx.debug(msg)   → Debug-level notification
    - ctx.warning(msg) → Warning notification
    - ctx.error(msg)   → Error notification
    - ctx.report_progress(current, total, msg) → Progress updates

    Workflow:
    1. Client calls this tool
    2. Server sends notifications during processing
    3. Client receives notifications in real-time
    4. Server returns final result

    Args:
        location: Location for weather query
        ctx: MCP Context for sending notifications (injected by FastMCP)

    Returns:
        str: Weather information

    Teaching Notes:
        - Context is automatically injected - just add it as a parameter
        - Notifications are fire-and-forget (no response expected)
        - Use for: progress bars, debug info, warnings, status updates
        - Client decides how to display notifications (log, UI, ignore)
    """
    # Send initial notification
    await ctx.info(f"🔍 Starting weather lookup for: {location}")

    # Report progress (step 1 of 3)
    await ctx.report_progress(1, 3, "Validating API credentials...")

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        await ctx.error("❌ SERPAPI_API_KEY not configured!")
        return "Error: SERPAPI_API_KEY environment variable not set."

    # Use debug_log helper - respects the toggle_debug_mode setting
    await debug_log(ctx, "✓ API key found")

    # Report progress (step 2 of 3)
    await ctx.report_progress(2, 3, "Querying weather service...")

    try:
        client = Client(api_key=api_key)
        params = {
            "q": f"What's the weather in {location}?",
            "hl": "en",
            "gl": "us"
        }

        await ctx.info(f"📡 Sending request to SerpAPI...")
        results = client.search(engine="google", params=params)

        # Report progress (step 3 of 3)
        await ctx.report_progress(3, 3, "Processing results...")

        if "error" in results:
            await ctx.warning(f"⚠️ API returned error: {results['error']}")
            return f"Error from SerpAPI: {results['error']}"

        answer_box = results.get("answer_box")
        if not answer_box:
            await ctx.warning(f"⚠️ No weather data in response for {location}")
            return f"No weather data found for {location}"

        temperature = answer_box.get('temperature', 'N/A')
        conditions = answer_box.get('weather', 'N/A')

        await ctx.info(f"✅ Successfully retrieved weather for {location}")

        return f"Weather in {location}: {temperature} and {conditions}"

    except Exception as e:
        await ctx.error(f"❌ Exception occurred: {str(e)}")
        return f"Error getting weather: {str(e)}"


# ==============================================================================
# SECTION 7: SERVER↔CLIENT SAMPLING
# ==============================================================================
# Sampling allows the SERVER to request LLM completions from the CLIENT
# This is POWERFUL: the server can leverage the client's LLM!

@mcp.tool(
    name="get_weather_advice",
    description="Get weather AND personalized advice using ctx.sample() to request LLM from client."
)
async def get_weather_advice(location: str, activity: str, ctx: Context) -> str:
    """
    Weather tool that demonstrates SAMPLING - server requesting LLM from client.

    SAMPLING (Available in FastMCP 2.0+):
    -------------------------------------
    ctx.sample() sends a request to the CLIENT's LLM for completion.
    The server provides data, the client provides intelligence!

    API: ctx.sample(messages, system_prompt=None, temperature=None, max_tokens=None)
    Returns: TextContent | ImageContent | AudioContent

    Why is this powerful?
    - Server focuses on data (weather API)
    - Client provides intelligence (LLM)
    - Best of both worlds!
    - Server doesn't need its own API keys!

    Workflow:
    1. Client calls this tool
    2. Server fetches weather data from SerpAPI
    3. Server calls ctx.sample() → Client's LLM generates advice
    4. Server combines data + advice
    5. Returns comprehensive response

    Client Support Requirements:
    - Claude Desktop: ✅ Supported
    - Pydantic AI: ✅ Supported
    - Custom clients: Need sampling_handler implementation
    - If client doesn't support sampling, tool falls back gracefully

    Args:
        location: Location for weather query
        activity: What the user wants to do (hiking, beach, etc.)
        ctx: MCP Context for sampling (injected by FastMCP)

    Returns:
        str: Weather data combined with LLM-generated advice

    Teaching Notes:
        - ctx.sample() sends a prompt to the CLIENT'S LLM
        - Server doesn't need its own LLM!
        - Use for: summarization, translation, personalization
        - Client controls which LLM is used and its parameters
        - Sampling requires client permission (security consideration)
    """
    # =================================================================
    # DEMONSTRATION: Using notifications to show sampling workflow
    # These ctx.info(), ctx.report_progress() calls push updates
    # to the client in real-time, making the process transparent.
    # =================================================================

    await ctx.info(f"🎯 Starting weather advice request for {activity} in {location}")
    await ctx.report_progress(0, 4, "Initializing weather advice request...")

    # Step 1: Get the weather data (server's job)
    await ctx.report_progress(1, 4, "Step 1/4: Validating API credentials...")
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        await ctx.error("❌ SERPAPI_API_KEY not configured!")
        return "Error: SERPAPI_API_KEY not set"

    try:
        await ctx.report_progress(2, 4, "Step 2/4: Fetching weather data from SerpAPI...")
        await ctx.info("🌐 Querying weather service...")

        client = Client(api_key=api_key)
        params = {"q": f"weather in {location}", "hl": "en", "gl": "us"}
        results = client.search(engine="google", params=params)

        answer_box = results.get("answer_box", {})
        temperature = answer_box.get('temperature', 'unknown')
        conditions = answer_box.get('weather', 'unknown')

        weather_data = f"Temperature: {temperature}, Conditions: {conditions}"

        await ctx.info(f"📊 Weather data retrieved: {weather_data}")

        # Step 2: Use SAMPLING to get advice from CLIENT'S LLM
        # ================================================================
        # ctx.sample() is available in FastMCP 2.0+ and sends a request
        # to the CLIENT's LLM for completion. The server provides data,
        # the client provides intelligence!
        #
        # API: ctx.sample(messages, system_prompt=None, temperature=None, max_tokens=None)
        # Returns: TextContent | ImageContent | AudioContent
        #
        # IMPORTANT: Client must support sampling capability
        # - Claude Desktop: Supported
        # - Pydantic AI: Supported
        # - Custom clients: Need sampling_handler implementation
        # ================================================================

        # Step 3: Use SAMPLING to get advice from CLIENT'S LLM
        await ctx.report_progress(3, 4, "Step 3/4: Consulting LLM for personalized advice...")
        await ctx.info("🤖 Just a sec, consulting the client's LLM via sampling...")
        # Use debug_log helper - respects the toggle_debug_mode setting
        await debug_log(ctx, "This is where the magic happens - server asks CLIENT for LLM help!")

        sampling_prompt = f"""
Given the following weather conditions:
Location: {location}
{weather_data}

The user wants to: {activity}

Provide brief, practical advice (2-3 sentences) about whether
this is a good idea and any precautions they should take.
"""

        try:
            # ctx.sample() asks the CLIENT's LLM to generate a response
            # The client controls which model is used!
            await ctx.info("⏳ Waiting for client LLM response...")
            advice_response = await ctx.sample(
                messages=sampling_prompt,
                max_tokens=200,
                temperature=0.7
            )

            # Extract the text from the sampling response
            # Returns TextContent, ImageContent, or AudioContent
            advice = advice_response.text if hasattr(advice_response, 'text') else str(advice_response)

            await ctx.info("✅ Got it! Client LLM returned personalized advice")

        except Exception as sampling_error:
            # Graceful fallback if client doesn't support sampling
            await ctx.warning(f"⚠️ Sampling not supported by client: {sampling_error}")
            await ctx.info("📝 Using fallback advice generation (no LLM)")
            advice = f"[Fallback - client doesn't support sampling] " \
                     f"Consider the {conditions} conditions and {temperature} temperature " \
                     f"when planning your {activity}."

        # Step 4: Combine server data + client LLM intelligence
        await ctx.report_progress(4, 4, "Step 4/4: Combining results...")
        await ctx.info("🔄 Combining weather data with personalized advice...")
        await ctx.info("✨ All done! Here's your weather advice:")

        return f"""
🌤️ Weather Report for {location}:
{weather_data}

💡 Advice for {activity}:
{advice}
"""

    except Exception as e:
        await ctx.error(f"❌ Error: {str(e)}")
        return f"Error getting weather advice: {str(e)}"


# ==============================================================================
# SERVER EXECUTION
# ==============================================================================

if __name__ == "__main__":
    """
    Main entry point for the MCP server.

    Execution Flow:
    ---------------
    1. Environment variables loaded (lines 33-38)
    2. MCP server created (line 45)
    3. Tools, resources, and prompts registered via decorators
    4. Server starts and listens for MCP protocol messages

    Running the Server:
    -------------------
    Command line:
        python stoopid_wx_server.py

    Or with MCP client configuration:
        {
          "mcpServers": {
            "weather": {
              "command": "python",
              "args": ["stoopid_wx_server.py"]
            }
          }
        }

    Server Lifecycle:
    -----------------
    - Startup: Registers all decorated functions with MCP protocol
    - Running: Responds to tool calls, resource requests, prompt queries
    - Shutdown: Graceful cleanup on interrupt (Ctrl+C)

    Teaching Notes:
        - mcp.run() is a blocking call - starts the server loop
        - Server communicates via stdin/stdout (MCP protocol)
        - FastMCP handles protocol details automatically
        - In production, add logging, metrics, and health checks
    """
    print("=" * 70)
    print("🌤️  Stoopid Weather Server - MCP Client Teaching Demo")
    print("=" * 70)
    print("\n📚 TEACHING SECTIONS:")
    print("-" * 40)
    print("Section 1-3: Basic MCP Primitives")
    print("   • Tool: get_weather(location)")
    print("   • Resource: file://readme")
    print("   • Resource: data://app-status")
    print("   • Prompt: hip_weather(location)")
    print()
    print("Section 4: Prompt + Resource Stacking")
    print("   • Prompt: weather_with_context(location)")
    print("   • Prompt: multi_resource()")
    print()
    print("Section 5: Roots (File System Boundaries)")
    print("   • Tool: get_roots() - query client for allowed directories")
    print("   • Tool: read_file_safe(path) - read file with roots enforcement")
    print("   • Tool: list_files(directory) - list directory with roots enforcement")
    print("   • Resource: file://config")
    print("   • Documentation in docstrings")
    print()
    print("Section 6: Server→Client Notifications")
    print("   • Tool: get_weather_with_logging(location)")
    print("   • Uses: ctx.info(), ctx.debug(), ctx.warning()")
    print("   • Uses: ctx.report_progress()")
    print()
    print("Section 7: Server↔Client Sampling")
    print("   • Tool: get_weather_advice(location, activity)")
    print("   • Uses: ctx.sample() to request LLM from client")
    print("-" * 40)
    print("\n✨ Server ready for MCP protocol messages\n")

    mcp.run()


"""
# extension to include forecast in the output
        forecast = answer_box.get('forecast', 'N/A')
        return f"Weather in {location}: {temperature} and {conditions} with forecast {forecast}"
"""