"""Configuration utilities for AI Tale core.

This module provides functions for loading and managing configuration settings
for the AI Tale core engine.
"""

import os
import logging
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "llm": {
        "provider": "mock",  # Default to mock provider for safety
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "image": {
        "provider": "mock",  # Default to mock provider for safety
        "model": "dall-e-3",
        "size": "1024x1024"
    },
    "story_templates": {
        "introduction": "Write an engaging introduction for a children's fairy tale about {{theme}}. The story is for children aged {{age_group}}. {{section_prompt}}",
        "conflict": "Write the conflict section of a children's fairy tale about {{theme}}. The story is for children aged {{age_group}}. {{section_prompt}}",
        "rising_action": "Write the rising action section of a children's fairy tale about {{theme}}. The story is for children aged {{age_group}}. {{section_prompt}}",
        "climax": "Write the climax section of a children's fairy tale about {{theme}}. The story is for children aged {{age_group}}. {{section_prompt}}",
        "resolution": "Write the resolution section of a children's fairy tale about {{theme}}. The story is for children aged {{age_group}}. {{section_prompt}}",
        "epilogue": "Write a brief epilogue for a children's fairy tale about {{theme}}. The story is for children aged {{age_group}}. {{section_prompt}}"
    },
    "default_story_params": {
        "theme": "adventure",
        "age_group": "6-8",
        "length": "medium"
    },
    "validation": {
        "banned_patterns": [
            "\\b(kill|murder|blood|gore|violent|suicide|drug|cocaine|heroin)\\b",
            "\\b(sex|sexual|naked|nude|explicit)\\b"
        ],
        "age_appropriate_rules": {
            "3-5": {
                "max_sentence_length": 10,
                "avoid_scary_content": True,
                "max_complexity_score": 30
            },
            "6-8": {
                "max_sentence_length": 15,
                "avoid_scary_content": True,
                "max_complexity_score": 50
            },
            "9-12": {
                "max_sentence_length": 20,
                "avoid_scary_content": False,
                "max_complexity_score": 70
            }
        }
    },
    "illustration": {
        "default_style": "children's book illustration",
        "default_num_illustrations": 5
    },
    "export": {
        "default_format": "html"
    }
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a file or use default configuration.

    Args:
        config_path: Path to the configuration file. If None, default config is used.

    Returns:
        Configuration dictionary.
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path:
        try:
            # Check if the config file exists
            if not os.path.exists(config_path):
                logger.warning(f"Configuration file not found: {config_path}. Using default configuration.")
                return config
            
            # Load configuration from file
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Merge with default configuration
            if file_config:
                config = _deep_merge(config, file_config)
                
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.warning("Using default configuration.")
    else:
        logger.info("No configuration file provided. Using default configuration.")
    
    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary.
        override: Dictionary to override base values.

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = _deep_merge(result[key], value)
        else:
            # Override or add the value
            result[key] = value
    
    return result


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to a file.

    Args:
        config: Configuration dictionary.
        config_path: Path where to save the configuration.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Save configuration to file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Saved configuration to {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")


def create_default_config(config_path: str) -> None:
    """Create a default configuration file.

    Args:
        config_path: Path where to save the default configuration.
    """
    save_config(DEFAULT_CONFIG, config_path)