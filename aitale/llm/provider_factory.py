"""LLM Provider Factory module for AI Tale.

This module provides factory functions for creating LLM providers and
implements provider fallback mechanisms for robust operation.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Type

from aitale.llm.providers.base import LLMProvider
from aitale.llm.providers import AVAILABLE_PROVIDERS

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory class for creating and managing LLM providers.
    
    This class handles the creation of LLM providers based on configuration
    and implements fallback mechanisms for robust operation.
    """
    
    @staticmethod
    def create_provider(config: Dict[str, Any]) -> LLMProvider:
        """Create an LLM provider based on configuration.

        Args:
            config: Configuration dictionary with provider settings.

        Returns:
            An instance of a class implementing the LLMProvider interface.

        Raises:
            ValueError: If the specified provider is not supported.
        """
        provider_type = config.get("provider", "openai").lower()
        
        if provider_type not in AVAILABLE_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {provider_type}. Supported providers: {', '.join(AVAILABLE_PROVIDERS.keys())}")
        
        provider_class = AVAILABLE_PROVIDERS[provider_type]
        
        try:
            # Validate the configuration before creating the provider
            validated_config = provider_class.validate_config(config)
            return provider_class(validated_config)
        except Exception as e:
            logger.error(f"Error creating {provider_type} provider: {e}")
            raise
    
    @staticmethod
    def create_provider_with_fallback(config: Dict[str, Any]) -> LLMProvider:
        """Create an LLM provider with fallback options for robust operation.

        This function attempts to create the primary provider specified in the
        configuration. If that fails, it will try fallback providers in order.

        Args:
            config: Configuration dictionary with provider settings.

        Returns:
            An instance of a class implementing the LLMProvider interface.

        Raises:
            ValueError: If all provider creation attempts fail.
        """
        # Get the primary provider type
        primary_provider = config.get("provider", "openai").lower()
        
        # Get fallback providers list
        fallback_providers = config.get("fallback_providers", [])
        
        # Try to create the primary provider
        try:
            return ProviderFactory.create_provider(config)
        except Exception as primary_error:
            logger.warning(f"Failed to create primary provider '{primary_provider}': {primary_error}")
            
            # If no fallbacks are specified, try mock as a last resort
            if not fallback_providers:
                fallback_providers = ["mock"]
            
            # Try each fallback provider
            errors = [f"{primary_provider}: {primary_error}"]
            for fallback in fallback_providers:
                try:
                    # Create a new config with the fallback provider
                    fallback_config = config.copy()
                    fallback_config["provider"] = fallback
                    
                    logger.info(f"Attempting to use fallback provider: {fallback}")
                    return ProviderFactory.create_provider(fallback_config)
                except Exception as fallback_error:
                    logger.warning(f"Failed to create fallback provider '{fallback}': {fallback_error}")
                    errors.append(f"{fallback}: {fallback_error}")
            
            # If all providers fail, raise an error with details
            error_details = "\n".join(errors)
            raise ValueError(f"Failed to create any LLM provider. Errors:\n{error_details}")


def get_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """Factory function to create an LLM provider based on configuration.

    This is a convenience function that uses ProviderFactory to create a provider.
    It supports robust operation with fallback providers if enabled in the config.

    Args:
        config: Configuration dictionary with provider settings.

    Returns:
        An instance of a class implementing the LLMProvider interface.

    Raises:
        ValueError: If provider creation fails.
    """
    # Check if fallback is enabled
    use_fallback = config.get("use_fallback", True)
    
    if use_fallback:
        return ProviderFactory.create_provider_with_fallback(config)
    else:
        return ProviderFactory.create_provider(config)