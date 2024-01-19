"""Export Engine module for AI Tale.

This module handles the export of illustrated stories to various formats,
including HTML, PDF, and other presentation formats.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import base64
import shutil

from jinja2 import Environment, FileSystemLoader, select_autoescape
from aitale.core.models import IllustratedStory, Story
from aitale.utils.config import load_config

logger = logging.getLogger(__name__)


class ExportEngine:
    """Exports illustrated stories to various formats.

    This class handles the conversion of IllustratedStory objects to
    various presentation formats, including HTML, PDF, and others.
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the ExportEngine.

        Args:
            config_path: Path to the configuration file. If not provided, default config will be used.
            config: Configuration dictionary. If provided, this will override config_path.
        """
        if config:
            self.config = config
        else:
            self.config = load_config(config_path)

        self.export_config = self.config.get("export", {})
        
        # Set up Jinja2 environment for templates
        template_dir = self.export_config.get("template_dir", os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "../templates"
        ))
        
        # Create template directory if it doesn't exist
        os.makedirs(template_dir, exist_ok=True)
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Create default templates if they don't exist
        self._ensure_default_templates(template_dir)

    def export_html(self, story: IllustratedStory, output_path: str) -> str:
        """Export an illustrated story to HTML format.

        Args:
            story: The IllustratedStory object to export.
            output_path: Path where the HTML file should be saved.

        Returns:
            Path to the exported HTML file.
        """
        # Ensure the output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Add extension if not present
        if not output_path.endswith(".html"):
            output_path = f"{output_path}.html"
        
        # Create assets directory for images
        assets_dir = os.path.join(output_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Save illustrations to assets directory and create image paths
        image_paths = []
        for i, illustration in enumerate(story.illustrations):
            image_filename = f"illustration_{i+1}.png"
            image_path = os.path.join(assets_dir, image_filename)
            
            # Handle different types of image_data
            if isinstance(illustration.image_data, bytes):
                # Binary image data
                with open(image_path, "wb") as f:
                    f.write(illustration.image_data)
            elif isinstance(illustration.image_data, str):
                if illustration.image_data.startswith("data:image"):
                    # Base64 encoded image
                    header, encoded = illustration.image_data.split(",", 1)
                    image_data = base64.b64decode(encoded)
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                elif os.path.isfile(illustration.image_data):
                    # File path
                    shutil.copy(illustration.image_data, image_path)
                else:
                    # Treat as raw image data
                    with open(image_path, "wb") as f:
                        f.write(illustration.image_data.encode("utf-8"))
            
            image_paths.append(os.path.join("assets", image_filename))
        
        # Prepare data for the template
        template_data = {
            "title": story.title,
            "sections": story.sections,
            "illustrations": [
                {
                    "image_path": image_path,
                    "section_index": illustration.section_index,
                    "prompt": illustration.prompt
                }
                for image_path, illustration in zip(image_paths, story.illustrations)
            ],
            "metadata": story.metadata
        }
        
        # Render the HTML template
        template = self.jinja_env.get_template("storybook.html")
        html_content = template.render(**template_data)
        
        # Write the HTML file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return output_path

    def export_pdf(self, story: IllustratedStory, output_path: str) -> str:
        """Export an illustrated story to PDF format.

        Args:
            story: The IllustratedStory object to export.
            output_path: Path where the PDF file should be saved.

        Returns:
            Path to the exported PDF file.

        Note:
            This method requires additional dependencies (e.g., weasyprint)
            which should be installed separately.
        """
        # Ensure the output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Add extension if not present
        if not output_path.endswith(".pdf"):
            output_path = f"{output_path}.pdf"
        
        # First export to HTML
        html_path = f"{os.path.splitext(output_path)[0]}_temp.html"
        self.export_html(story, html_path)
        
        try:
            # Try to import weasyprint for PDF conversion
            from weasyprint import HTML
            
            # Convert HTML to PDF
            HTML(html_path).write_pdf(output_path)
            
            # Remove temporary HTML file
            os.remove(html_path)
            
            return output_path
        except ImportError:
            logger.warning("WeasyPrint not installed. Cannot export to PDF. Using HTML export instead.")
            # If weasyprint is not available, return the HTML path
            return html_path

    def _ensure_default_templates(self, template_dir: str) -> None:
        """Ensure that default templates exist in the template directory.

        Args:
            template_dir: Path to the template directory.
        """
        # Create default HTML template if it doesn't exist
        html_template_path = os.path.join(template_dir, "storybook.html")
        if not os.path.exists(html_template_path):
            default_html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - AI Tale</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .story-container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .story-section {
            margin-bottom: 30px;
        }
        .story-section h2 {
            color: #3498db;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
            margin-top: 30px;
        }
        .illustration {
            text-align: center;
            margin: 30px 0;
        }
        .illustration img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .illustration-caption {
            font-style: italic;
            color: #7f8c8d;
            margin-top: 10px;
            font-size: 0.9em;
        }
        .metadata {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-top: 40px;
            font-size: 0.9em;
        }
        .metadata h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .metadata-item {
            margin-bottom: 5px;
        }
        footer {
            text-align: center;
            margin-top: 40px;
            color: #7f8c8d;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="story-container">
        <h1>{{ title }}</h1>
        
        {% for section in sections %}
        <div class="story-section">
            {% if loop.first %}
            {% else %}
            <h2>{{ section.type|title }}</h2>
            {% endif %}
            
            <div class="section-content">
                {{ section.content|replace('\n', '<br>')|safe }}
            </div>
            
            {% for illustration in illustrations %}
                {% if illustration.section_index == loop.parent.index0 %}
                <div class="illustration">
                    <img src="{{ illustration.image_path }}" alt="Illustration for {{ section.type }}">
                    <div class="illustration-caption">{{ illustration.prompt|truncate(100) }}</div>
                </div>
                {% endif %}
            {% endfor %}
        </div>
        {% endfor %}
        
        <div class="metadata">
            <h3>About this story</h3>
            {% if metadata.theme %}
            <div class="metadata-item"><strong>Theme:</strong> {{ metadata.theme }}</div>
            {% endif %}
            {% if metadata.age_group %}
            <div class="metadata-item"><strong>Age Group:</strong> {{ metadata.age_group }}</div>
            {% endif %}
            {% if metadata.protagonist %}
            <div class="metadata-item"><strong>Protagonist:</strong> {{ metadata.protagonist }}</div>
            {% endif %}
            {% if metadata.illustration_style %}
            <div class="metadata-item"><strong>Illustration Style:</strong> {{ metadata.illustration_style }}</div>
            {% endif %}
        </div>
    </div>
    
    <footer>
        <p>Generated by <a href="https://aitale.tech/" target="_blank">AI Tale</a> - Bringing imagination to life with AI</p>
    </footer>
</body>
</html>
"""
            os.makedirs(os.path.dirname(html_template_path), exist_ok=True)
            with open(html_template_path, "w", encoding="utf-8") as f:
                f.write(default_html_template)
        
        # Create CSS template if needed
        css_template_path = os.path.join(template_dir, "storybook.css")
        if not os.path.exists(css_template_path):
            default_css_template = """
/* AI Tale Storybook Styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f9f9f9;
}

h1 {
    color: #2c3e50;
    text-align: center;
    margin-bottom: 30px;
    font-size: 2.5em;
}

.story-container {
    background-color: white;
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.story-section {
    margin-bottom: 30px;
}

.story-section h2 {
    color: #3498db;
    border-bottom: 2px solid #ecf0f1;
    padding-bottom: 10px;
    margin-top: 30px;
}

.illustration {
    text-align: center;
    margin: 30px 0;
}

.illustration img {
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.illustration-caption {
    font-style: italic;
    color: #7f8c8d;
    margin-top: 10px;
    font-size: 0.9em;
}

.metadata {
    background-color: #ecf0f1;
    padding: 15px;
    border-radius: 5px;
    margin-top: 40px;
    font-size: 0.9em;
}

.metadata h3 {
    margin-top: 0;
    color: #2c3e50;
}

.metadata-item {
    margin-bottom: 5px;
}

footer {
    text-align: center;
    margin-top: 40px;
    color: #7f8c8d;
    font-size: 0.8em;
}
"""
            os.makedirs(os.path.dirname(css_template_path), exist_ok=True)
            with open(css_template_path, "w", encoding="utf-8") as f:
                f.write(default_css_template)