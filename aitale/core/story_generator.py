"""Story Generator module for AI Tale.

This module handles the generation of fairy tales using large language models.
It provides functionality to create stories based on various parameters such as
theme, age group, protagonist, and length.
"""

import logging
import os
from typing import Dict, Any, Optional, List

import yaml

from aitale.core.models import Story, StorySection
from aitale.llm.provider import LLMProvider, get_llm_provider
from aitale.utils.config import load_config

logger = logging.getLogger(__name__)


class StoryGenerator:
    """Generates fairy tales using large language models.

    This class handles the generation of fairy tales by interfacing with
    language models. It supports customization of story parameters and
    ensures the generated content is appropriate for the target audience.
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the StoryGenerator.

        Args:
            config_path: Path to the configuration file. If not provided, default config will be used.
            config: Configuration dictionary. If provided, this will override config_path.
        """
        if config:
            self.config = config
        else:
            self.config = load_config(config_path)

        self.llm_provider = get_llm_provider(self.config.get("llm", {}))
        self.story_templates = self.config.get("story_templates", {})
        self.default_params = self.config.get("default_story_params", {})

    def generate(self, 
                theme: Optional[str] = None, 
                age_group: Optional[str] = None, 
                protagonist: Optional[str] = None, 
                length: Optional[str] = None, 
                **kwargs) -> Story:
        """Generate a fairy tale based on the provided parameters.

        Args:
            theme: The theme of the story (e.g., "adventure", "friendship").
            age_group: Target age group (e.g., "3-5", "6-8", "9-12").
            protagonist: Main character type (e.g., "princess", "dragon", "wizard").
            length: Story length ("short", "medium", "long").
            **kwargs: Additional parameters for story generation.

        Returns:
            A Story object containing the generated fairy tale.
        """
        # Merge default parameters with provided ones
        params = self.default_params.copy()
        if theme:
            params["theme"] = theme
        if age_group:
            params["age_group"] = age_group
        if protagonist:
            params["protagonist"] = protagonist
        if length:
            params["length"] = length
        params.update(kwargs)

        # Validate parameters
        self._validate_params(params)

        # Generate story structure
        structure = self._generate_structure(params)

        # Generate each section of the story
        sections = []
        for section_type, section_prompt in structure.items():
            section_content = self._generate_section(section_type, section_prompt, params)
            sections.append(StorySection(type=section_type, content=section_content))

        # Create and return the story object
        return Story(
            title=self._generate_title(sections, params),
            sections=sections,
            metadata={
                "theme": params.get("theme"),
                "age_group": params.get("age_group"),
                "protagonist": params.get("protagonist"),
                "length": params.get("length"),
                **kwargs
            }
        )

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate the story generation parameters.

        Args:
            params: Dictionary of story parameters.

        Raises:
            ValueError: If any parameters are invalid.
        """
        # Check required parameters
        required_params = ["theme", "age_group"]
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")

        # Validate age group format
        age_group = params.get("age_group")
        if age_group and not self._is_valid_age_group(age_group):
            raise ValueError(f"Invalid age group format: {age_group}. Expected format: 'X-Y'")

        # Validate length
        valid_lengths = ["short", "medium", "long", None]
        if params.get("length") not in valid_lengths:
            raise ValueError(f"Invalid length: {params.get('length')}. Expected one of: {valid_lengths[:-1]}")

    def _is_valid_age_group(self, age_group: str) -> bool:
        """Check if the age group format is valid.

        Args:
            age_group: Age group string to validate.

        Returns:
            True if the age group is valid, False otherwise.
        """
        try:
            if "-" not in age_group:
                return False
            min_age, max_age = map(int, age_group.split("-"))
            return 0 <= min_age <= max_age <= 18
        except (ValueError, TypeError):
            return False

    def _generate_structure(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Generate the structure of the story.

        Args:
            params: Dictionary of story parameters.

        Returns:
            Dictionary mapping section types to section prompts.
        """
        # Determine story length and corresponding structure
        length = params.get("length", "medium")
        if length == "short":
            return {
                "introduction": "Introduce the main character and setting",
                "conflict": "Present a simple problem or challenge",
                "resolution": "Resolve the conflict with a happy ending"
            }
        elif length == "long":
            return {
                "introduction": "Introduce the main character, setting, and background",
                "character_development": "Develop the character's personality and motivations",
                "conflict": "Present a complex problem or challenge",
                "rising_action": "Show the character's journey and obstacles",
                "climax": "Present the turning point of the story",
                "falling_action": "Show the consequences of the climax",
                "resolution": "Resolve the conflict with a meaningful ending",
                "epilogue": "Provide a glimpse of life after the main events"
            }
        else:  # medium (default)
            return {
                "introduction": "Introduce the main character and setting",
                "conflict": "Present a problem or challenge",
                "rising_action": "Show the character's journey and obstacles",
                "climax": "Present the turning point of the story",
                "resolution": "Resolve the conflict with a satisfying ending"
            }

    def _generate_section(self, section_type: str, section_prompt: str, params: Dict[str, Any]) -> str:
        """Generate a section of the story.

        Args:
            section_type: Type of the section (e.g., "introduction", "conflict").
            section_prompt: Prompt describing what the section should contain.
            params: Dictionary of story parameters.

        Returns:
            Generated content for the section.
        """
        # Get the appropriate template for this section type
        template = self.story_templates.get(section_type, self.story_templates.get("default", ""))

        # If no template is found, create a basic one
        if not template:
            template = f"Write a {section_type} for a fairy tale where {{section_prompt}}. "
            template += "The story is about {{theme}} and is for children aged {{age_group}}. "
            if params.get("protagonist"):
                template += "The main character is a {{protagonist}}. "

        # Format the template with the parameters
        formatted_template = template.format(
            section_prompt=section_prompt,
            **params
        )

        # Generate content using the LLM
        response = self.llm_provider.generate_text(formatted_template)
        return response.strip()

    def _generate_title(self, sections: List[StorySection], params: Dict[str, Any]) -> str:
        """Generate a title for the story.

        Args:
            sections: List of story sections.
            params: Dictionary of story parameters.

        Returns:
            Generated title for the story.
        """
        # Create a prompt for title generation
        introduction = next((s.content for s in sections if s.type == "introduction"), "")
        prompt = f"Generate a captivating title for a fairy tale with the following introduction: '{introduction[:200]}...'"
        
        if params.get("protagonist"):
            prompt += f" The main character is a {params['protagonist']}."
        
        if params.get("theme"):
            prompt += f" The theme is {params['theme']}."

        # Generate title using the LLM
        title = self.llm_provider.generate_text(prompt, max_tokens=20)
        
        # Clean up the title
        title = title.strip().strip('"').strip("'").strip()
        
        return title