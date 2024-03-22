"""LLM Provider module for AI Tale.

This module provides the interface for language model providers and
factory functions for creating provider instances.
"""

import logging
from typing import Dict, Any

# Re-export the base provider interface
from aitale.llm.providers.base import LLMProvider

# Re-export the provider factory function
from aitale.llm.provider_factory import get_llm_provider

# Re-export all provider implementations for backward compatibility
from aitale.llm.providers import (
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    CohereProvider,
    MockProvider,
    AVAILABLE_PROVIDERS
)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI language model provider.

    This class implements the LLMProvider interface for OpenAI's language models,
    including GPT-3.5, GPT-4, and other models available through the OpenAI API.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the OpenAI provider.

        Args:
            config: Configuration dictionary with OpenAI-specific settings.
        """
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI package is not installed. Install it with 'pip install openai'.")

        self.config = config
        
        # Set up API key
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Provide it in the config or set the OPENAI_API_KEY environment variable.")
        
        # Initialize the client
        self.client = openai.OpenAI(api_key=api_key)
        
        # Set default model
        self.default_model = config.get("model", "gpt-3.5-turbo")
        
        # Set default parameters
        self.default_params = {
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 1000),
            "top_p": config.get("top_p", 1.0),
            "frequency_penalty": config.get("frequency_penalty", 0.0),
            "presence_penalty": config.get("presence_penalty", 0.0)
        }

    def generate_text(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate text using OpenAI's language models.

        Args:
            prompt: The prompt to send to the language model.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional OpenAI-specific parameters.

        Returns:
            Generated text as a string.
        """
        # Prepare parameters
        params = self.default_params.copy()
        if max_tokens:
            params["max_tokens"] = max_tokens
        params.update(kwargs)
        
        # Use the chat endpoint with a single user message for text generation
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.default_model),
                messages=messages,
                **params
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            return f"Error: {str(e)}"

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate a response based on a conversation history using OpenAI's chat models.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional OpenAI-specific parameters.

        Returns:
            Generated response as a string.
        """
        # Prepare parameters
        params = self.default_params.copy()
        if max_tokens:
            params["max_tokens"] = max_tokens
        params.update(kwargs)
        
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.default_model),
                messages=messages,
                **params
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating chat response with OpenAI: {e}")
            return f"Error: {str(e)}"


class AnthropicProvider(LLMProvider):
    """Anthropic language model provider.

    This class implements the LLMProvider interface for Anthropic's Claude models.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Anthropic provider.

        Args:
            config: Configuration dictionary with Anthropic-specific settings.
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package is not installed. Install it with 'pip install anthropic'.")

        self.config = config
        
        # Set up API key
        api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found. Provide it in the config or set the ANTHROPIC_API_KEY environment variable.")
        
        # Initialize the client
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Set default model
        self.default_model = config.get("model", "claude-2")
        
        # Set default parameters
        self.default_params = {
            "temperature": config.get("temperature", 0.7),
            "max_tokens_to_sample": config.get("max_tokens", 1000),
            "top_p": config.get("top_p", 1.0),
        }

    def generate_text(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate text using Anthropic's language models.

        Args:
            prompt: The prompt to send to the language model.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional Anthropic-specific parameters.

        Returns:
            Generated text as a string.
        """
        # Prepare parameters
        params = self.default_params.copy()
        if max_tokens:
            params["max_tokens_to_sample"] = max_tokens
        params.update(kwargs)
        
        try:
            response = self.client.completions.create(
                model=kwargs.get("model", self.default_model),
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                **params
            )
            return response.completion
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {e}")
            return f"Error: {str(e)}"

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate a response based on a conversation history using Anthropic's models.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional Anthropic-specific parameters.

        Returns:
            Generated response as a string.
        """
        # Prepare parameters
        params = self.default_params.copy()
        if max_tokens:
            params["max_tokens_to_sample"] = max_tokens
        params.update(kwargs)
        
        # Convert messages to Anthropic's format
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user" or role == "human":
                prompt += f"{anthropic.HUMAN_PROMPT} {content}"
            elif role == "assistant" or role == "ai":
                prompt += f"{anthropic.AI_PROMPT} {content}"
            else:
                # Skip system messages or convert them to human messages
                if role == "system":
                    prompt += f"{anthropic.HUMAN_PROMPT} [System instruction: {content}]"
        
        # Add final AI prompt
        prompt += anthropic.AI_PROMPT
        
        try:
            response = self.client.completions.create(
                model=kwargs.get("model", self.default_model),
                prompt=prompt,
                **params
            )
            return response.completion
        except Exception as e:
            logger.error(f"Error generating chat response with Anthropic: {e}")
            return f"Error: {str(e)}"


class MockProvider(LLMProvider):
    """Mock language model provider for testing.

    This class implements a simple mock provider that returns predefined
    responses for testing purposes without requiring API access.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the mock provider.

        Args:
            config: Configuration dictionary with mock provider settings.
        """
        self.config = config
        self.responses = config.get("responses", {})
        self.default_response = config.get("default_response", "This is a mock response from the AI.")

    def generate_text(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate mock text based on the provided prompt.

        Args:
            prompt: The prompt to match against predefined responses.
            max_tokens: Ignored in the mock provider.
            **kwargs: Ignored in the mock provider.

        Returns:
            Predefined response as a string.
        """
        # Check if we have a matching response for this prompt
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return response
        
        # Return default response
        return self.default_response

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate a mock response based on a conversation history.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            max_tokens: Ignored in the mock provider.
            **kwargs: Ignored in the mock provider.

        Returns:
            Predefined response as a string.
        """
        # Get the last user message
        last_user_message = ""
        for message in reversed(messages):
            if message["role"] == "user":
                last_user_message = message["content"]
                break
        
        # Check if we have a matching response for this message
        for pattern, response in self.responses.items():
            if pattern in last_user_message:
                return response
        
        # Return default response
        return self.default_response


def get_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """Factory function to create an LLM provider based on configuration.

    Args:
        config: Configuration dictionary with provider settings.

    Returns:
        An instance of a class implementing the LLMProvider interface.

    Raises:
        ValueError: If the specified provider is not supported.
    """
    provider_type = config.get("provider", "openai").lower()
    
    if provider_type == "openai":
        return OpenAIProvider(config)
    elif provider_type == "anthropic":
        return AnthropicProvider(config)
    elif provider_type == "mock":
        return MockProvider(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_type}")