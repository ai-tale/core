"""Google Provider module for AI Tale.

This module implements the LLM provider interface for Google's Gemini (formerly PaLM) models.
"""

import logging
import os
from typing import Dict, Any, Optional, List

from aitale.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class GoogleProvider(LLMProvider):
    """Google language model provider.

    This class implements the LLMProvider interface for Google's Gemini models,
    providing access to Google's advanced language models.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Google provider.

        Args:
            config: Configuration dictionary with Google-specific settings.
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI package is not installed. Install it with 'pip install google-generativeai'")

        # Store the Google module for later use
        self.genai = genai
        
        # Validate and normalize configuration
        self.config = self.validate_config(config)
        
        # Configure the API
        self.genai.configure(api_key=self.config["api_key"])
        
        # Set default model
        self.default_model = self.config.get("model", "gemini-pro")
        
        # Set default parameters
        self.default_params = {
            "temperature": self.config.get("temperature", 0.7),
            "max_output_tokens": self.config.get("max_tokens", 1000),
            "top_p": self.config.get("top_p", 0.95),
            "top_k": self.config.get("top_k", 40),
        }

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the Google provider configuration.
        
        Args:
            config: Configuration dictionary with Google settings.
            
        Returns:
            Validated and normalized configuration dictionary.
            
        Raises:
            ValueError: If the configuration is invalid.
        """
        validated_config = config.copy()
        
        # Check for API key
        api_key = validated_config.get("api_key") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Provide it in the config or set the GOOGLE_API_KEY environment variable.")
        validated_config["api_key"] = api_key
        
        # Validate model
        model = validated_config.get("model", "gemini-pro")
        # List of supported Gemini models
        supported_models = [
            "gemini-pro", "gemini-pro-vision", "gemini-ultra"
        ]
        
        if model not in supported_models:
            logger.warning(f"Model '{model}' may not be supported by Google. Supported models: {', '.join(supported_models)}")
        
        validated_config["model"] = model
        
        # Validate temperature
        temperature = validated_config.get("temperature", 0.7)
        if not 0 <= temperature <= 1:
            logger.warning(f"Temperature {temperature} is outside recommended range [0, 1]. Clamping to valid range.")
            temperature = max(0, min(temperature, 1))
        validated_config["temperature"] = temperature
        
        return validated_config

    def generate_text(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate text using Google's language models.

        Args:
            prompt: The prompt to send to the language model.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional Google-specific parameters.

        Returns:
            Generated text as a string.
        """
        # Prepare parameters
        params = self.default_params.copy()
        if max_tokens:
            params["max_output_tokens"] = max_tokens
        params.update(kwargs)
        
        try:
            # Get the model
            model = self.genai.GenerativeModel(kwargs.get("model", self.default_model))
            
            # Generate content
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": params.pop("temperature", 0.7),
                    "max_output_tokens": params.pop("max_output_tokens", 1000),
                    "top_p": params.pop("top_p", 0.95),
                    "top_k": params.pop("top_k", 40),
                },
                **params
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with Google: {e}")
            return f"Error: {str(e)}"

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate a response based on a conversation history using Google's models.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional Google-specific parameters.

        Returns:
            Generated response as a string.
        """
        # Prepare parameters
        params = self.default_params.copy()
        if max_tokens:
            params["max_output_tokens"] = max_tokens
        params.update(kwargs)
        
        try:
            # Get the model
            model = self.genai.GenerativeModel(kwargs.get("model", self.default_model))
            
            # Start a chat session
            chat = model.start_chat()
            
            # Convert messages to Google's format and add them to the chat
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "user" or role == "human":
                    # Add user message to chat history
                    chat.send_message(content)
                elif role == "assistant" or role == "ai":
                    # For assistant messages, we need to simulate them in the history
                    # This is a limitation of the Google API
                    chat._history.append({"role": "model", "parts": [content]})
                elif role == "system":
                    # System messages are handled as user messages with a special prefix
                    chat.send_message(f"[System instruction: {content}]")
            
            # Generate the response with the final parameters
            response = chat.send_message(
                "",  # Empty message to get the next response based on history
                generation_config={
                    "temperature": params.pop("temperature", 0.7),
                    "max_output_tokens": params.pop("max_output_tokens", 1000),
                    "top_p": params.pop("top_p", 0.95),
                    "top_k": params.pop("top_k", 40),
                },
                **params
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error generating chat response with Google: {e}")
            return f"Error: {str(e)}"