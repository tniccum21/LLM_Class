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
- SERPAPI_API_KEY environment variable (stored in secrets.env)
- Required packages: fastmcp, serpapi, python-dotenv

"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv
from fastmcp import FastMCP
from serpapi import Client

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

# Load environment variables from secrets.env
# This keeps sensitive API keys out of version control
# Pattern: ALWAYS use .env type files for secrets, never hardcode credentials
env_path = Path(__file__).parent / "secrets.env"
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
        return "Error: SERPAPI_API_KEY environment variable not set. Please set it in secrets.env"

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
# Resources are PASSIVE data providers - static information the LLM can read
# Think of resources as "nouns" - things the LLM can ACCESS

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
    # Use Path for platform-independent file operations
    readme_path = Path(__file__).parent / "README_stoopid_wx_server.md"

    if readme_path.exists():
        return readme_path.read_text()
    else:
        return "README file not found."


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
    return {
        "status": "ok",
        "uptime": 12345,  # In production, calculate actual uptime
        "hosted_at": mcp.settings.host
    }


# ==============================================================================
# MCP PROMPT IMPLEMENTATION
# ==============================================================================
# Prompts are TEMPLATE generators - reusable instruction patterns for LLMs
# Think of prompts as "prompt engineering as code"

@mcp.prompt
def what_is_the_hip_weather_report(location: str) -> str:
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
        Client lists available prompts → sees "what_is_the_hip_weather_report"
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

@mcp.prompt
def weather_with_context(location: str) -> str:
    """
    Generate a weather prompt that references available resources.

    This demonstrates PROMPT + RESOURCE STACKING:
    - The prompt instructs the LLM to use BOTH:
      1. The get_weather TOOL to fetch live data
      2. The data://app-status RESOURCE to check service health

    STACKING PATTERN:
    -----------------
    Prompt (instructions) + Resource (context) + Tool (action) = Rich interaction

    Workflow when this prompt is used:
    1. Client requests this prompt with location
    2. Prompt tells LLM: "Check status resource, then call weather tool"
    3. LLM reads data://app-status resource
    4. LLM calls get_weather(location) tool
    5. LLM synthesizes both into cohesive response

    Args:
        location: The location for the weather report

    Returns:
        str: A prompt that instructs the LLM to combine resources and tools

    Teaching Notes:
        - Prompts can reference OTHER MCP primitives by name
        - This creates declarative workflows: "do X, then Y, combine Z"
        - The LLM orchestrates the actual execution
        - Stacking reduces client-side complexity
    """
    return f"""
    Before providing the weather report, please:

    1. FIRST: Check the application status by reading the 'data://app-status' resource
       - If status is not "ok", warn the user about potential data quality issues

    2. THEN: Get the current weather for {location} using the get_weather tool

    3. FINALLY: Provide a comprehensive weather report that includes:
       - The current conditions from the tool
       - A note about data freshness based on the app status
       - A friendly recommendation for the day

    Combine all information into a cohesive, helpful response.
    """


@mcp.prompt
def multi_resource_prompt() -> str:
    """
    A prompt demonstrating how to reference MULTIPLE resources.

    This shows advanced stacking where the LLM should:
    - Read the README resource for context about capabilities
    - Read the app-status resource for current state
    - Synthesize into a system overview

    Use Case: System documentation and status queries

    Returns:
        str: Instructions for multi-resource synthesis

    Teaching Notes:
        - Prompts can reference ANY number of resources
        - Order matters - process foundational context first
        - LLM handles the orchestration automatically
    """
    return """
    Please provide a comprehensive overview of this weather service:

    1. Read the 'file://readme' resource to understand:
       - What this service does
       - How to use it
       - Available features

    2. Read the 'data://app-status' resource to report:
       - Current operational status
       - Uptime information
       - Server health

    3. Combine into a professional system status report that a user
       could use to understand both WHAT this service does and
       WHETHER it's currently operational.
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

# Example resource that would benefit from roots configuration
@mcp.resource(uri="file://config")
def get_config_file() -> str:
    """
    Demonstrate a resource that reads from the file system.

    In production with roots configured:
    - This would only work if the config file is within a declared root
    - MCP client enforces the boundary

    Returns:
        str: Config file contents or error message

    Teaching Notes:
        - This resource assumes roots are properly configured
        - Without roots, this is a security risk
        - Always validate paths are within expected boundaries
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
# Import the logging utilities

from mcp.server.fastmcp import Context

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

    await ctx.debug("✓ API key found")

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
    await ctx.info(f"🎯 Getting weather advice for {activity} in {location}")

    # Step 1: Get the weather data (server's job)
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY not set"

    try:
        client = Client(api_key=api_key)
        params = {"q": f"weather in {location}", "hl": "en", "gl": "us"}
        results = client.search(engine="google", params=params)

        answer_box = results.get("answer_box", {})
        temperature = answer_box.get('temperature', 'unknown')
        conditions = answer_box.get('weather', 'unknown')

        weather_data = f"Temperature: {temperature}, Conditions: {conditions}"

        await ctx.info(f"📊 Weather data: {weather_data}")

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

        await ctx.info("🤖 Requesting advice from client LLM via sampling...")

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
            advice_response = await ctx.sample(
                messages=sampling_prompt,
                max_tokens=200,
                temperature=0.7
            )

            # Extract the text from the sampling response
            # Returns TextContent, ImageContent, or AudioContent
            advice = advice_response.text if hasattr(advice_response, 'text') else str(advice_response)

            await ctx.info("✅ Received advice from client LLM via sampling")

        except Exception as sampling_error:
            # Graceful fallback if client doesn't support sampling
            await ctx.warning(f"⚠️ Sampling not supported by client: {sampling_error}")
            advice = f"[Fallback - client doesn't support sampling] " \
                     f"Consider the {conditions} conditions and {temperature} temperature " \
                     f"when planning your {activity}."

        # Step 3: Combine server data + client LLM intelligence
        return f"""
🌤️ Weather Report for {location}:
{weather_data}

💡 Advice for {activity}:
{advice}
"""

    except Exception as e:
        await ctx.error(f"Error: {str(e)}")
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
    print("   • Prompt: what_is_the_hip_weather_report(location)")
    print()
    print("Section 4: Prompt + Resource Stacking")
    print("   • Prompt: weather_with_context(location)")
    print("   • Prompt: multi_resource_prompt()")
    print()
    print("Section 5: Roots (File System Boundaries)")
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