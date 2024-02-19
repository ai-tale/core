"""Illustration Integrator module for AI Tale.

This module handles the integration of AI-generated illustrations with fairy tales.
It provides functionality to generate and incorporate illustrations based on
story content and parameters.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

from aitale.core.models import Story, IllustratedStory, Illustration
from aitale.image.provider import ImageProvider, get_image_provider
from aitale.utils.config import load_config

logger = logging.getLogger(__name__)


class IllustrationIntegrator:
    """Integrates AI-generated illustrations with fairy tales.

    This class handles the generation and integration of illustrations
    for fairy tales. It interfaces with image generation services and
    ensures the illustrations match the story content and style.
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the IllustrationIntegrator.

        Args:
            config_path: Path to the configuration file. If not provided, default config will be used.
            config: Configuration dictionary. If provided, this will override config_path.
        """
        if config:
            self.config = config
        else:
            self.config = load_config(config_path)

        self.image_provider = get_image_provider(self.config.get("image", {}))
        self.illustration_config = self.config.get("illustration", {})
        self.default_style = self.illustration_config.get("default_style", "children's book illustration")
        self.default_num_illustrations = self.illustration_config.get("default_num_illustrations", 5)

    def illustrate(self, 
                  story: Story, 
                  style: Optional[str] = None, 
                  num_illustrations: Optional[int] = None,
                  **kwargs) -> IllustratedStory:
        """Generate illustrations for a story and integrate them.

        Args:
            story: The Story object to illustrate.
            style: Illustration style (e.g., "watercolor", "digital art").
            num_illustrations: Number of illustrations to generate.
            **kwargs: Additional parameters for illustration generation.

        Returns:
            An IllustratedStory object containing the story with integrated illustrations.
        """
        # Set default values if not provided
        style = style or self.default_style
        num_illustrations = num_illustrations or self._determine_num_illustrations(story)

        # Determine key scenes for illustration
        illustration_points = self._identify_illustration_points(story, num_illustrations)

        # Generate illustrations for each point
        illustrations = []
        for section_idx, content_snippet in illustration_points:
            prompt = self._create_illustration_prompt(content_snippet, style, story.metadata)
            image_data = self.image_provider.generate_image(prompt, **kwargs)
            
            illustration = Illustration(
                image_data=image_data,
                prompt=prompt,
                section_index=section_idx,
                metadata={
                    "style": style,
                    **kwargs
                }
            )
            illustrations.append(illustration)

        # Create and return the illustrated story
        return IllustratedStory(
            title=story.title,
            sections=story.sections,
            illustrations=illustrations,
            metadata={
                **story.metadata,
                "illustration_style": style,
                "num_illustrations": len(illustrations)
            }
        )

    def _determine_num_illustrations(self, story: Story) -> int:
        """Determine the appropriate number of illustrations based on story length.

        Args:
            story: The Story object to analyze.

        Returns:
            Recommended number of illustrations.
        """
        # Count total words in the story
        total_words = sum(len(section.content.split()) for section in story.sections)
        
        # Determine number of illustrations based on word count
        if total_words < 500:  # Short story
            return min(3, self.default_num_illustrations)
        elif total_words < 1500:  # Medium story
            return min(5, self.default_num_illustrations)
        else:  # Long story
            return min(8, self.default_num_illustrations)

    def _identify_illustration_points(self, story: Story, num_illustrations: int) -> List[Tuple[int, str]]:
        """Identify the best points in the story for illustrations.

        Args:
            story: The Story object to analyze.
            num_illustrations: Number of illustrations to generate.

        Returns:
            List of tuples containing (section_index, content_snippet) for illustration.
        """
        # Always illustrate the introduction
        points = []
        intro_idx = next((i for i, s in enumerate(story.sections) if s.type == "introduction"), 0)
        points.append((intro_idx, self._extract_snippet(story.sections[intro_idx].content)))
        
        # Always illustrate the climax if it exists
        climax_idx = next((i for i, s in enumerate(story.sections) if s.type == "climax"), None)
        if climax_idx is not None:
            points.append((climax_idx, self._extract_snippet(story.sections[climax_idx].content)))
        
        # Distribute remaining illustrations evenly
        remaining_points = num_illustrations - len(points)
        if remaining_points > 0 and len(story.sections) > 2:
            # Skip sections already selected for illustration
            available_sections = [i for i in range(len(story.sections)) 
                                if i != intro_idx and i != climax_idx]
            
            # If we need more illustrations than available sections, some sections will get multiple illustrations
            if remaining_points <= len(available_sections):
                # Select evenly spaced sections
                step = len(available_sections) // remaining_points
                for i in range(0, len(available_sections), step):
                    if len(points) < num_illustrations:  # Check to avoid exceeding requested number
                        section_idx = available_sections[i]
                        points.append((section_idx, self._extract_snippet(story.sections[section_idx].content)))
            else:
                # Add all available sections
                for section_idx in available_sections:
                    points.append((section_idx, self._extract_snippet(story.sections[section_idx].content)))
                
                # If we still need more, add additional snippets from longer sections
                remaining = num_illustrations - len(points)
                if remaining > 0:
                    # Find the longest sections
                    section_lengths = [(i, len(s.content.split())) 
                                     for i, s in enumerate(story.sections)]
                    section_lengths.sort(key=lambda x: x[1], reverse=True)
                    
                    for section_idx, _ in section_lengths[:remaining]:
                        # Extract a different snippet from the section
                        content = story.sections[section_idx].content
                        if len(content.split()) > 100:  # Only if the section is long enough
                            # Extract from the middle or end if we already have the beginning
                            if (section_idx, self._extract_snippet(content)) in points:
                                middle_snippet = self._extract_snippet(content, position="middle")
                                points.append((section_idx, middle_snippet))
        
        # Sort by section index to maintain story flow
        points.sort(key=lambda x: x[0])
        
        return points[:num_illustrations]  # Ensure we don't exceed the requested number

    def _extract_snippet(self, content: str, position: str = "beginning", length: int = 150) -> str:
        """Extract a snippet from the content for illustration prompt.

        Args:
            content: The text content to extract from.
            position: Where to extract from ("beginning", "middle", "end").
            length: Approximate length of the snippet in characters.

        Returns:
            A text snippet suitable for illustration.
        """
        words = content.split()
        if len(words) <= length // 5:  # If content is already short
            return content
            
        if position == "beginning":
            return " ".join(words[:length // 5])
        elif position == "end":
            return " ".join(words[-(length // 5):])
        else:  # middle
            middle_idx = len(words) // 2
            start_idx = max(0, middle_idx - (length // 10))
            end_idx = min(len(words), middle_idx + (length // 10))
            return " ".join(words[start_idx:end_idx])

    def _create_illustration_prompt(self, content_snippet: str, style: str, metadata: Dict[str, Any]) -> str:
        """Create a prompt for generating an illustration.

        Args:
            content_snippet: Text snippet to base the illustration on.
            style: Illustration style.
            metadata: Story metadata for context.

        Returns:
            A prompt string for the image generation service.
        """
        # Start with the content snippet
        prompt = f"Create a {style} illustration for a children's fairy tale with the following scene: {content_snippet}"
        
        # Add age-appropriate guidance
        age_group = metadata.get("age_group")
        if age_group:
            prompt += f" The illustration should be appropriate for children aged {age_group}."
        
        # Add theme and protagonist information if available
        if "theme" in metadata:
            prompt += f" The story theme is {metadata['theme']}."
            
        if "protagonist" in metadata:
            prompt += f" The main character is a {metadata['protagonist']}."
        
        # Add style-specific instructions
        if "watercolor" in style.lower():
            prompt += " Use soft colors and gentle brushstrokes."
        elif "digital" in style.lower():
            prompt += " Use vibrant colors and clean lines."
        elif "cartoon" in style.lower():
            prompt += " Use expressive characters and simple backgrounds."
            
        # Safety instructions
        prompt += " The illustration must be child-friendly, non-violent, and appropriate for the specified age group."
        
        return prompt