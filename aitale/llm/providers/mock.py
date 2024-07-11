"""Mock Provider module for AI Tale.

This module implements a mock LLM provider for testing and development.
"""

import logging
from typing import Dict, Any, Optional, List

from aitale.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


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
        # Validate and normalize configuration
        self.config = self.validate_config(config)
        self.responses = self.config.get("responses", {})
        self.default_response = self.config.get("default_response", "This is a mock response from the AI.")

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the mock provider configuration.
        
        Args:
            config: Configuration dictionary with mock provider settings.
            
        Returns:
            Validated and normalized configuration dictionary.
        """
        validated_config = config.copy()
        
        # Ensure responses is a dictionary
        if "responses" in validated_config and not isinstance(validated_config["responses"], dict):
            logger.warning("Mock provider 'responses' should be a dictionary. Using empty dictionary instead.")
            validated_config["responses"] = {}
        
        # Ensure default_response is a string
        if "default_response" in validated_config and not isinstance(validated_config["default_response"], str):
            logger.warning("Mock provider 'default_response' should be a string. Using default value instead.")
            validated_config["default_response"] = "This is a mock response from the AI."
        
        return validated_config

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
            if message["role"] in ["user", "human"]:
                last_user_message = message["content"]
                break
        
        # Check if we have a matching response for this message
        for pattern, response in self.responses.items():
            if pattern in last_user_message:
                return response
        
        # Return default response
        return self.default_response