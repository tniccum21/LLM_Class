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
    print("🌤️  Stoopid Weather Server Starting...")
    print("=" * 70)
    print("🔧 Registered MCP primitives:")
    print("   • Tool: get_weather(location)")
    print("   • Resource: file://readme")
    print("   • Resource: data://app-status")
    print("   • Prompt: what_is_the_weather_report(location)")
    print("\n✨ Server ready for MCP protocol messages\n")

    mcp.run()


"""
# extension to include forecast in the output
        forecast = answer_box.get('forecast', 'N/A')
        return f"Weather in {location}: {temperature} and {conditions} with forecast {forecast}"
"""