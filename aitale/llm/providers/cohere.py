"""Cohere Provider module for AI Tale.

This module implements the LLM provider interface for Cohere's language models.
"""

import logging
import os
from typing import Dict, Any, Optional, List

from aitale.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class CohereProvider(LLMProvider):
    """Cohere language model provider.

    This class implements the LLMProvider interface for Cohere's language models,
    providing access to models like Command and Generate.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Cohere provider.

        Args:
            config: Configuration dictionary with Cohere-specific settings.
        """
        try:
            import cohere
        except ImportError:
            raise ImportError("Cohere package is not installed. Install it with 'pip install cohere'")

        # Validate and normalize configuration
        self.config = self.validate_config(config)
        
        # Initialize the client
        self.client = cohere.Client(api_key=self.config["api_key"])
        
        # Set default model
        self.default_model = self.config.get("model", "command")
        
        # Set default parameters
        self.default_params = {
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 1000),
            "p": self.config.get("top_p", 0.75),
            "k": self.config.get("top_k", 0),
            "frequency_penalty": self.config.get("frequency_penalty", 0.0),
            "presence_penalty": self.config.get("presence_penalty", 0.0)
        }

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the Cohere provider configuration.
        
        Args:
            config: Configuration dictionary with Cohere settings.
            
        Returns:
            Validated and normalized configuration dictionary.
            
        Raises:
            ValueError: If the configuration is invalid.
        """
        validated_config = config.copy()
        
        # Check for API key
        api_key = validated_config.get("api_key") or os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError("Cohere API key not found. Provide it in the config or set the COHERE_API_KEY environment variable.")
        validated_config["api_key"] = api_key
        
        # Validate model
        model = validated_config.get("model", "command")
        # List of supported Cohere models
        supported_models = [
            "command", "command-light", "command-nightly", "command-light-nightly",
            "generate", "generate-light"
        ]
        
        if model not in supported_models:
            logger.warning(f"Model '{model}' may not be supported by Cohere. Supported models: {', '.join(supported_models)}")
        
        validated_config["model"] = model
        
        # Validate temperature
        temperature = validated_config.get("temperature", 0.7)
        if not 0 <= temperature <= 5:
            logger.warning(f"Temperature {temperature} is outside recommended range [0, 5]. Clamping to valid range.")
            temperature = max(0, min(temperature, 5))
        validated_config["temperature"] = temperature
        
        return validated_config

    def generate_text(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate text using Cohere's language models.

        Args:
            prompt: The prompt to send to the language model.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional Cohere-specific parameters.

        Returns:
            Generated text as a string.
        """
        # Prepare parameters
        params = self.default_params.copy()
        if max_tokens:
            params["max_tokens"] = max_tokens
        params.update(kwargs)
        
        try:
            # Determine which API to use based on the model
            model = kwargs.get("model", self.default_model)
            
            if model.startswith("command"):
                # Use the chat API for Command models
                response = self.client.chat(
                    message=prompt,
                    model=model,
                    temperature=params.pop("temperature", 0.7),
                    max_tokens=params.pop("max_tokens", 1000),
                    p=params.pop("p", 0.75),
                    k=params.pop("k", 0),
                    **params
                )
                return response.text
            else:
                # Use the generate API for Generate models
                response = self.client.generate(
                    prompt=prompt,
                    model=model,
                    temperature=params.pop("temperature", 0.7),
                    max_tokens=params.pop("max_tokens", 1000),
                    p=params.pop("p", 0.75),
                    k=params.pop("k", 0),
                    frequency_penalty=params.pop("frequency_penalty", 0.0),
                    presence_penalty=params.pop("presence_penalty", 0.0),
                    **params
                )
                return response.generations[0].text
        except Exception as e:
            logger.error(f"Error generating text with Cohere: {e}")
            return f"Error: {str(e)}"

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate a response based on a conversation history using Cohere's models.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional Cohere-specific parameters.

        Returns:
            Generated response as a string.
        """
        # Prepare parameters
        params = self.default_params.copy()
        if max_tokens:
            params["max_tokens"] = max_tokens
        params.update(kwargs)
        
        try:
            # Convert messages to Cohere's chat format
            chat_history = []
            system_message = None
            user_message = None
            
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    system_message = content
                elif role == "user" or role == "human":
                    # If we already have a user message, add the previous exchange to history
                    if user_message and len(chat_history) > 0 and chat_history[-1]["role"] == "CHATBOT":
                        chat_history.append({"role": "USER", "message": user_message})
                    user_message = content
                elif role == "assistant" or role == "ai":
                    if user_message:
                        chat_history.append({"role": "USER", "message": user_message})
                        chat_history.append({"role": "CHATBOT", "message": content})
                        user_message = None
                    else:
                        # If there's no preceding user message, just add to history
                        chat_history.append({"role": "CHATBOT", "message": content})
            
            # Determine which API to use based on the model
            model = kwargs.get("model", self.default_model)
            
            if model.startswith("command"):
                # Use the chat API for Command models
                response = self.client.chat(
                    message=user_message or "",  # Use the last user message or empty string
                    chat_history=chat_history,
                    model=model,
                    preamble=system_message,  # Use system message as preamble if available
                    temperature=params.pop("temperature", 0.7),
                    max_tokens=params.pop("max_tokens", 1000),
                    p=params.pop("p", 0.75),
                    k=params.pop("k", 0),
                    **params
                )
                return response.text
            else:
                # For Generate models, we need to format the conversation as a single prompt
                formatted_prompt = ""
                if system_message:
                    formatted_prompt += f"System: {system_message}\n\n"
                
                for entry in chat_history:
                    if entry["role"] == "USER":
                        formatted_prompt += f"User: {entry['message']}\n"
                    else:  # CHATBOT
                        formatted_prompt += f"Assistant: {entry['message']}\n"
                
                if user_message:
                    formatted_prompt += f"User: {user_message}\n"
                formatted_prompt += "Assistant: "
                
                response = self.client.generate(
                    prompt=formatted_prompt,
                    model=model,
                    temperature=params.pop("temperature", 0.7),
                    max_tokens=params.pop("max_tokens", 1000),
                    p=params.pop("p", 0.75),
                    k=params.pop("k", 0),
                    frequency_penalty=params.pop("frequency_penalty", 0.0),
                    presence_penalty=params.pop("presence_penalty", 0.0),
                    **params
                )
                return response.generations[0].text
        except Exception as e:
            logger.error(f"Error generating chat response with Cohere: {e}")
            return f"Error: {str(e)}"