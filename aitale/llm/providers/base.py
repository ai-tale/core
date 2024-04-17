"""Base LLM Provider module for AI Tale.

This module defines the interface for language model providers that all
provider implementations must adhere to.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for language model providers.

    This class defines the interface that all LLM providers must implement.
    It provides methods for generating text and handling provider-specific
    configuration and authentication.
    """

    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate text based on the provided prompt.

        Args:
            prompt: The prompt to send to the language model.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Generated text as a string.
        """
        pass

    @abstractmethod
    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate a response based on a conversation history.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Generated response as a string.
        """
        pass
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the provider configuration.
        
        Args:
            config: Configuration dictionary with provider settings.
            
        Returns:
            Validated and normalized configuration dictionary.
            
        Raises:
            ValueError: If the configuration is invalid.
        """
        # Base implementation just returns the config as-is
        # Provider implementations should override this to add validation
        return config