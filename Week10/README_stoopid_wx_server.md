# Stoopid Weather Server

![1763732184171](image/README_stoopid_wx_server/1763732184171.png)

A simple MCP (Model Context Protocol) server that provides weather information using SerpAPI's Google Search integration.

## Overview

This server exposes weather data through MCP tools, prompts, and resources, making it easy for AI assistants to fetch current weather conditions for any location.

## Features

- **Weather Tool**: Fetch weather data in human-readable text format
- **Weather Prompts**: Pre-configured weather reporting prompts
- **Documentation Resource**: Access server README documentation via MCP
- **Server Status**: Application status resource with uptime tracking

## Prerequisites

- Python 3.11+
- SerpAPI account and API key
- Required packages:
  - `fastmcp`
  - `serpapi`
  - `python-dotenv`

## Installation

1. Install dependencies:

```bash
pip install fastmcp serpapi python-dotenv
```

2. Create a `secrets.env` file in the same directory:

```env
SERPAPI_API_KEY=your_api_key_here
```

Get your SerpAPI key from: https://serpapi.com/

3. Configure LM Studio to use this MCP server:

Add the following to your LM Studio MCP configuration file (typically found in LM Studio settings under "MCP Servers"):

```json

```

**Important:**

- Replace `/absolute/path/to/` with the actual full path to your project directory (both places)
- Use `fastmcp` command from your `.venv/bin/` directory where it's installed
- The API key is automatically loaded from `secrets.env` in the same directory as the script

## Usage

### Running the Server

```bash
fastmcp run stoopid_wx_server.py
```

Or from the virtual environment:

```bash
source /path/to/.venv/bin/activate
fastmcp run stoopid_wx_server.py
```

The server will start and expose the following capabilities via MCP.

### Available Tools

#### `get_weather`

Returns human-readable weather information as text. The serpapi actually returns the google search weather "Answer Box" with a lot of info.

![1763731959779](image/README_stoopid_wx_server/1763731959779.png)

**Parameters:**

- `location` (str): City or location name

**Returns:**

```
Weather in Paris: 72°F and Partly Cloudy
```

**Example:**

```
"What's the weather in Paris?"
"Tell me the temperature in New York"
```

### Available Resources

#### `file://readme`

Returns the complete README documentation for this server.

**Returns:**

```
Complete README.md content as text
```

**Usage:**
Access the server's documentation directly through MCP to understand available features, configuration, and usage examples.

#### `data://app-status`

Application status with metadata.

**Returns:**

```json
{
  "status": "ok",
  "uptime": 12345,
  "hosted at": "localhost:8000"
}
```

### Available Prompts

#### `what_is_the_weather_report`

Creates a weather reporter prompt for a given location.

**Parameters:**

- `location` (str): Location for weather report

**Returns:**

```
You are a weather reporter. Weather report for Paris?
```

## Error Handling

The server handles various error cases:

- **Missing API Key**: Returns error message prompting to set `SERPAPI_API_KEY`
- **SerpAPI Errors**: Returns API error details
- **No Data Found**: Returns message when location has no weather data
- **General Exceptions**: Catches and reports unexpected errors

## Architecture

```
stoopid_wx_server.py
├── Environment Setup
│   └── Load SERPAPI_API_KEY from secrets.env
├── Tools
│   └── get_weather (text format)
├── Resources
│   ├── file://readme (README documentation)
│   └── data://app-status (server status)
└── Prompts
    └── what_is_the_weather_report
```

## MCP Integration

This server implements the Model Context Protocol, allowing AI assistants to:

1. Discover available weather tools automatically
2. Call tools with natural language requests
3. Receive structured or unstructured weather data
4. Access server resources and prompts

## Example Integration

When connected to an MCP-compatible AI assistant:

```
User: "What's the weather like in Tokyo?"
Assistant: [Calls get_weather tool with location="Tokyo"]
Server: "Weather in Tokyo: 68°F and Clear"

User: "Show me the server documentation"
Assistant: [Accesses file://readme resource]
Server: [Returns complete README content]
```

## Limitations

- Weather data accuracy depends on SerpAPI's Google Search results
- Requires active internet connection
- Subject to SerpAPI rate limits and quotas
- Temperature format depends on Google's default (typically Fahrenheit)

## Development

### File Structure

```
Week09/
├── stoopid_wx_server.py    # Main server implementation
├── secrets.env             # API key configuration (gitignored)
└── README_stoopid_wx_server.md
```

### Testing

Test the server using an MCP client or by running:

```bash
python stoopid_wx_server.py
```

The server will listen for MCP requests on the default FastMCP port.

## License

Educational use - Part of Apps with AI course materials

## Credits

- Built with [FastMCP](https://github.com/jlowin/fastmcp)
- Weather data via [SerpAPI](https://serpapi.com/)
