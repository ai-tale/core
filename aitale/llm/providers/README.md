# LLM Providers for AI Tale

This package contains implementations of various Large Language Model (LLM) providers that can be used with AI Tale for story generation and content validation.

## Available Providers

The following LLM providers are currently supported:

- **OpenAI** (`openai`): Provides access to GPT models like GPT-3.5 and GPT-4
- **Anthropic** (`anthropic`): Provides access to Claude models
- **Google** (`google`): Provides access to Gemini (formerly PaLM) models
- **Cohere** (`cohere`): Provides access to Cohere's Command and Generate models
- **Mock** (`mock`): A mock provider for testing without API access

## Configuration

Each provider requires specific configuration settings. Here's an example configuration for each provider:

### OpenAI

```yaml
llm:
  provider: openai
  api_key: "your_openai_api_key_here"  # Or use OPENAI_API_KEY environment variable
  model: "gpt-4"  # Default: gpt-3.5-turbo
  temperature: 0.7
  max_tokens: 1000
  top_p: 1.0
  frequency_penalty: 0.0
  presence_penalty: 0.0
```

### Anthropic

```yaml
llm:
  provider: anthropic
  api_key: "your_anthropic_api_key_here"  # Or use ANTHROPIC_API_KEY environment variable
  model: "claude-3-opus-20240229"  # Default: claude-2
  temperature: 0.7
  max_tokens: 1000
  top_p: 1.0
```

### Google (Gemini)

```yaml
llm:
  provider: google
  api_key: "your_google_api_key_here"  # Or use GOOGLE_API_KEY environment variable
  model: "gemini-pro"  # Default: gemini-pro
  temperature: 0.7
  max_tokens: 1000
  top_p: 0.95
  top_k: 40
```

### Cohere

```yaml
llm:
  provider: cohere
  api_key: "your_cohere_api_key_here"  # Or use COHERE_API_KEY environment variable
  model: "command"  # Default: command
  temperature: 0.7
  max_tokens: 1000
  p: 0.75  # top_p
  k: 0  # top_k
  frequency_penalty: 0.0
  presence_penalty: 0.0
```

### Mock

```yaml
llm:
  provider: mock
  default_response: "This is a mock response from the AI."
  responses:
    "keyword1": "Response for prompts containing keyword1"
    "keyword2": "Response for prompts containing keyword2"
```

## Fallback Mechanism

AI Tale supports a fallback mechanism to ensure robust operation. If the primary provider fails, the system can automatically try alternative providers.

```yaml
llm:
  provider: openai  # Primary provider
  use_fallback: true  # Enable fallback mechanism
  fallback_providers: ["anthropic", "google", "mock"]  # Providers to try in order
  # ... other configuration ...
```

## Adding a New Provider

To add a new LLM provider:

1. Create a new file in the `providers` directory (e.g., `newprovider.py`)
2. Implement a class that inherits from `LLMProvider` and implements all required methods
3. Add the provider to `__init__.py` in the `AVAILABLE_PROVIDERS` dictionary

Example implementation template:

```python
from typing import Dict, Any, Optional, List
from aitale.llm.providers.base import LLMProvider

class NewProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        # Initialize the provider with configuration
        self.config = self.validate_config(config)
        # ... provider-specific initialization ...

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        # Validate and normalize configuration
        validated_config = config.copy()
        # ... validation logic ...
        return validated_config

    def generate_text(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        # Generate text based on prompt
        # ... implementation ...

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, **kwargs) -> str:
        # Generate response based on conversation history
        # ... implementation ...
```

## Error Handling

All providers implement robust error handling to ensure that failures are properly logged and don't crash the application. When an error occurs, providers will log the error and return an error message as the response.

## Best Practices

1. **API Keys**: Always use environment variables for API keys instead of hardcoding them in configuration files
2. **Fallback Providers**: Configure fallback providers to ensure robust operation
3. **Model Selection**: Choose appropriate models based on your needs and budget
4. **Error Handling**: Monitor logs for provider errors and adjust configuration as needed