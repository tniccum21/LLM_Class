from anthropic import Anthropic, APIError, AuthenticationError, NotFoundError, RateLimitError
from anthropic.types import Message


class ClaudeAPIError(Exception):
    """Wrapper for Claude API errors with user-friendly messages."""
    pass


class Claude:
    def __init__(self, model: str):
        self.client = Anthropic()
        self.model = model

    def add_user_message(self, messages: list, message):
        user_message = {
            "role": "user",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message):
        assistant_message = {
            "role": "assistant",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(assistant_message)

    def text_from_message(self, message: Message):
        return "\n".join(
            [block.text for block in message.content if block.type == "text"]
        )

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> Message:
        params = {
            "model": self.model,
            "max_tokens": 8000,
            "messages": messages,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
        }

        if thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        if tools:
            params["tools"] = tools

        if system:
            params["system"] = system

        try:
            message = self.client.messages.create(**params)
            return message
        except NotFoundError as e:
            # Model name typo or invalid model
            raise ClaudeAPIError(
                f"❌ Model not found: '{self.model}'\n"
                f"   Check CLAUDE_MODEL in secrets_client.env\n"
                f"   Valid models: claude-sonnet-4-20250514, claude-haiku-3-5-20241022, etc."
            ) from e
        except AuthenticationError as e:
            raise ClaudeAPIError(
                f"❌ Authentication failed\n"
                f"   Check ANTHROPIC_API_KEY in secrets_client.env\n"
                f"   Get your key at: https://console.anthropic.com/"
            ) from e
        except RateLimitError as e:
            raise ClaudeAPIError(
                f"❌ Rate limit exceeded\n"
                f"   Wait a moment and try again, or check your API usage limits."
            ) from e
        except APIError as e:
            raise ClaudeAPIError(
                f"❌ API error: {e.message}\n"
                f"   Status: {e.status_code}"
            ) from e
