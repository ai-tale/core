"""OpenAI Provider module for AI Tale.

This module implements the LLM provider interface for OpenAI's language models.
"""

import logging
import os
from typing import Dict, Any, Optional, List

from aitale.llm.providers.base import LLMProvider

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
            raise ImportError("OpenAI package is not installed. Install it with 'pip install openai'")

        # Validate and normalize configuration
        self.config = self.validate_config(config)
        
        # Initialize the client
        self.client = openai.OpenAI(api_key=self.config["api_key"])
        
        # Set default model
        self.default_model = self.config.get("model", "gpt-3.5-turbo")
        
        # Set default parameters
        self.default_params = {
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 1000),
            "top_p": self.config.get("top_p", 1.0),
            "frequency_penalty": self.config.get("frequency_penalty", 0.0),
            "presence_penalty": self.config.get("presence_penalty", 0.0)
        }

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the OpenAI provider configuration.
        
        Args:
            config: Configuration dictionary with OpenAI settings.
            
        Returns:
            Validated and normalized configuration dictionary.
            
        Raises:
            ValueError: If the configuration is invalid.
        """
        validated_config = config.copy()
        
        # Check for API key
        api_key = validated_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Provide it in the config or set the OPENAI_API_KEY environment variable.")
        validated_config["api_key"] = api_key
        
        # Validate model
        model = validated_config.get("model", "gpt-3.5-turbo")
        # Add model validation logic here if needed
        validated_config["model"] = model
        
        # Validate temperature
        temperature = validated_config.get("temperature", 0.7)
        if not 0 <= temperature <= 2:
            logger.warning(f"Temperature {temperature} is outside recommended range [0, 2]. Clamping to valid range.")
            temperature = max(0, min(temperature, 2))
        validated_config["temperature"] = temperature
        
        return validated_config

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