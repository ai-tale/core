"""Data models for AI Tale core components.

This module defines the data structures used throughout the AI Tale core engine,
including Story, StorySection, Illustration, and IllustratedStory models.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import json
import os
from datetime import datetime


@dataclass
class StorySection:
    """Represents a section of a story.

    Attributes:
        type: The type of section (e.g., "introduction", "conflict", "resolution").
        content: The text content of the section.
    """
    type: str
    content: str


@dataclass
class Illustration:
    """Represents an illustration for a story.

    Attributes:
        image_data: The binary image data or a reference to the image file.
        prompt: The prompt used to generate the illustration.
        section_index: The index of the story section this illustration belongs to.
        metadata: Additional metadata about the illustration.
    """
    image_data: Union[bytes, str]
    prompt: str
    section_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Story:
    """Represents a generated fairy tale.

    Attributes:
        title: The title of the story.
        sections: List of StorySection objects that make up the story.
        metadata: Additional metadata about the story.
    """
    title: str
    sections: List[StorySection]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_text(self) -> str:
        """Convert the story to plain text format.

        Returns:
            The story as a formatted text string.
        """
        text = f"{self.title}\n\n"
        for section in self.sections:
            text += f"{section.content}\n\n"
        return text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the story to a dictionary.

        Returns:
            Dictionary representation of the story.
        """
        return {
            "title": self.title,
            "sections": [
                {"type": section.type, "content": section.content}
                for section in self.sections
            ],
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }
    
    def save(self, file_path: str, format: str = "json") -> str:
        """Save the story to a file.

        Args:
            file_path: Path where the story should be saved.
            format: File format ("json" or "txt").

        Returns:
            The path to the saved file.

        Raises:
            ValueError: If an unsupported format is specified.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Add extension if not present
        if not file_path.endswith(f".{format}"):
            file_path = f"{file_path}.{format}"
        
        if format.lower() == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        elif format.lower() == "txt":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.to_text())
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'txt'.")
            
        return file_path
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Story":
        """Create a Story object from a dictionary.

        Args:
            data: Dictionary containing story data.

        Returns:
            A new Story object.
        """
        sections = [
            StorySection(type=section["type"], content=section["content"])
            for section in data.get("sections", [])
        ]
        
        metadata = data.get("metadata", {})
        
        return cls(
            title=data.get("title", "Untitled Story"),
            sections=sections,
            metadata=metadata
        )
    
    @classmethod
    def load(cls, file_path: str) -> "Story":
        """Load a story from a file.

        Args:
            file_path: Path to the story file (JSON format).

        Returns:
            A Story object.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file isn't valid JSON.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls.from_dict(data)


@dataclass
class IllustratedStory(Story):
    """Represents a fairy tale with illustrations.

    Attributes:
        title: The title of the story.
        sections: List of StorySection objects that make up the story.
        illustrations: List of Illustration objects for the story.
        metadata: Additional metadata about the story.
    """
    illustrations: List[Illustration] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the illustrated story to a dictionary.

        Returns:
            Dictionary representation of the illustrated story.
        """
        story_dict = super().to_dict()
        
        # Add illustrations data
        story_dict["illustrations"] = [
            {
                "prompt": ill.prompt,
                "section_index": ill.section_index,
                "metadata": ill.metadata,
                # For image_data, store a reference or convert to base64 if it's binary
                "image_reference": ill.image_data if isinstance(ill.image_data, str) else None
            }
            for ill in self.illustrations
        ]
        
        return story_dict
    
    def export(self, output_path: str, format: str = "html") -> str:
        """Export the illustrated story to various formats.

        Args:
            output_path: Base path for the exported files.
            format: Export format ("html", "json", or "pdf").

        Returns:
            Path to the exported file.

        Raises:
            ValueError: If an unsupported format is specified.
        """
        # Ensure the directory exists
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Add extension if not present
        if not output_path.endswith(f".{format}"):
            output_path = f"{output_path}.{format}"
        
        if format.lower() == "json":
            # Save as JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "html":
            # Create an HTML storybook
            from aitale.core.export_engine import ExportEngine
            export_engine = ExportEngine()
            output_path = export_engine.export_html(self, output_path)
        
        elif format.lower() == "pdf":
            # Create a PDF storybook
            from aitale.core.export_engine import ExportEngine
            export_engine = ExportEngine()
            output_path = export_engine.export_pdf(self, output_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'html', 'json', or 'pdf'.")
            
        return output_path