"""Anthropic Provider module for AI Tale.

This module implements the LLM provider interface for Anthropic's Claude models.
"""

import logging
import os
from typing import Dict, Any, Optional, List

from aitale.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


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
            raise ImportError("Anthropic package is not installed. Install it with 'pip install anthropic'")

        # Store the Anthropic module for later use
        self.anthropic_module = anthropic
        
        # Validate and normalize configuration
        self.config = self.validate_config(config)
        
        # Initialize the client
        self.client = anthropic.Anthropic(api_key=self.config["api_key"])
        
        # Set default model
        self.default_model = self.config.get("model", "claude-2")
        
        # Set default parameters
        self.default_params = {
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens_to_sample": self.config.get("max_tokens", 1000),
            "top_p": self.config.get("top_p", 1.0),
        }

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the Anthropic provider configuration.
        
        Args:
            config: Configuration dictionary with Anthropic settings.
            
        Returns:
            Validated and normalized configuration dictionary.
            
        Raises:
            ValueError: If the configuration is invalid.
        """
        validated_config = config.copy()
        
        # Check for API key
        api_key = validated_config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found. Provide it in the config or set the ANTHROPIC_API_KEY environment variable.")
        validated_config["api_key"] = api_key
        
        # Validate model
        model = validated_config.get("model", "claude-2")
        # List of supported Claude models
        supported_models = [
            "claude-2", "claude-2.0", "claude-2.1", 
            "claude-instant-1", "claude-instant-1.2",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
        ]
        
        if model not in supported_models:
            logger.warning(f"Model '{model}' may not be supported by Anthropic. Supported models: {', '.join(supported_models)}")
        
        validated_config["model"] = model
        
        # Validate temperature
        temperature = validated_config.get("temperature", 0.7)
        if not 0 <= temperature <= 1:
            logger.warning(f"Temperature {temperature} is outside recommended range [0, 1]. Clamping to valid range.")
            temperature = max(0, min(temperature, 1))
        validated_config["temperature"] = temperature
        
        return validated_config

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
            # Check if we're using Claude 3 models which use a different API
            if self.default_model.startswith("claude-3") or kwargs.get("model", "").startswith("claude-3"):
                # Claude 3 uses the messages API
                response = self.client.messages.create(
                    model=kwargs.get("model", self.default_model),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=params.pop("max_tokens_to_sample", 1000),
                    temperature=params.pop("temperature", 0.7),
                    **params
                )
                return response.content[0].text
            else:
                # Older Claude models use the completions API
                response = self.client.completions.create(
                    model=kwargs.get("model", self.default_model),
                    prompt=f"{self.anthropic_module.HUMAN_PROMPT} {prompt}{self.anthropic_module.AI_PROMPT}",
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
        
        try:
            # Check if we're using Claude 3 models which use a different API
            if self.default_model.startswith("claude-3") or kwargs.get("model", "").startswith("claude-3"):
                # Convert messages to Anthropic's format
                anthropic_messages = []
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    
                    if role == "user" or role == "human":
                        anthropic_messages.append({"role": "user", "content": content})
                    elif role == "assistant" or role == "ai":
                        anthropic_messages.append({"role": "assistant", "content": content})
                    elif role == "system":
                        # Claude 3 supports system messages
                        anthropic_messages.append({"role": "system", "content": content})
                
                response = self.client.messages.create(
                    model=kwargs.get("model", self.default_model),
                    messages=anthropic_messages,
                    max_tokens=params.pop("max_tokens_to_sample", 1000),
                    temperature=params.pop("temperature", 0.7),
                    **params
                )
                return response.content[0].text
            else:
                # Convert messages to Anthropic's format for older models
                prompt = ""
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    
                    if role == "user" or role == "human":
                        prompt += f"{self.anthropic_module.HUMAN_PROMPT} {content}"
                    elif role == "assistant" or role == "ai":
                        prompt += f"{self.anthropic_module.AI_PROMPT} {content}"
                    else:
                        # Handle system messages for older Claude models
                        if role == "system":
                            prompt += f"{self.anthropic_module.HUMAN_PROMPT} [System instruction: {content}]"
                
                # Add final AI prompt
                prompt += self.anthropic_module.AI_PROMPT
                
                response = self.client.completions.create(
                    model=kwargs.get("model", self.default_model),
                    prompt=prompt,
                    **params
                )
                return response.completion
        except Exception as e:
            logger.error(f"Error generating chat response with Anthropic: {e}")
            return f"Error: {str(e)}"