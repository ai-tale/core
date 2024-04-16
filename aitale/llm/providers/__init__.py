"""LLM Providers package for AI Tale.

This package contains implementations of various LLM providers that can be used
with AI Tale for story generation and content validation.
"""

# Import provider implementations for easy access
from aitale.llm.providers.openai import OpenAIProvider
from aitale.llm.providers.anthropic import AnthropicProvider
from aitale.llm.providers.google import GoogleProvider
from aitale.llm.providers.cohere import CohereProvider
from aitale.llm.providers.mock import MockProvider

# Define available providers for easy reference
AVAILABLE_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "cohere": CohereProvider,
    "mock": MockProvider
}