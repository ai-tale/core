"""Command-line interface for AI Tale core.

This module provides a command-line interface for interacting with the AI Tale
core engine, allowing users to generate stories and illustrations from the terminal.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional, List

import click

from aitale.core import StoryGenerator, IllustrationIntegrator, ContentValidator
from aitale.utils.config import load_config, create_default_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
def main():
    """AI Tale - Fairy tale generation with AI."""
    pass


@main.command()
@click.option("--theme", "-t", help="Theme of the story (e.g., adventure, friendship)")
@click.option("--age", "-a", help="Target age group (e.g., 3-5, 6-8, 9-12)")
@click.option("--protagonist", "-p", help="Main character type (e.g., princess, dragon, wizard)")
@click.option("--length", "-l", type=click.Choice(["short", "medium", "long"]), help="Length of the story")
@click.option("--config", "-c", help="Path to configuration file")
@click.option("--output", "-o", help="Output file path")
@click.option("--format", "-f", type=click.Choice(["json", "txt"]), default="json", help="Output format")
@click.option("--validate/--no-validate", default=True, help="Validate the story for appropriateness")
def generate(theme, age, protagonist, length, config, output, format, validate):
    """Generate a fairy tale based on the provided parameters."""
    try:
        # Load configuration
        config_dict = load_config(config)
        
        # Initialize story generator
        story_gen = StoryGenerator(config=config_dict)
        
        # Generate the story
        click.echo("Generating story...")
        story = story_gen.generate(
            theme=theme,
            age_group=age,
            protagonist=protagonist,
            length=length
        )
        
        # Validate the story if requested
        if validate:
            click.echo("Validating story...")
            validator = ContentValidator(config=config_dict)
            is_valid, issues = validator.validate_story(story)
            
            if not is_valid:
                click.echo("Story validation found issues:")
                for issue in issues:
                    click.echo(f"- {issue['type']}: {issue['detail']}")
                
                click.echo("Fixing issues...")
                story = validator.fix_issues(story, issues)
                click.echo("Issues fixed.")
        
        # Save the story
        if output:
            file_path = story.save(output, format=format)
            click.echo(f"Story saved to {file_path}")
        else:
            # Print the story to the console
            click.echo("\n" + "=" * 50)
            click.echo(f"Title: {story.title}")
            click.echo("=" * 50)
            
            for section in story.sections:
                if section.type != "introduction":  # Don't print section type for introduction
                    click.echo(f"\n--- {section.type.upper()} ---")
                click.echo(section.content)
            
            click.echo("\n" + "=" * 50)
        
        click.echo("Story generation completed successfully.")
        
    except Exception as e:
        logger.error(f"Error generating story: {e}", exc_info=True)
        click.echo(f"Error: {str(e)}")
        sys.exit(1)


@main.command()
@click.option("--story", "-s", required=True, help="Path to the story file (JSON format)")
@click.option("--style", help="Illustration style (e.g., watercolor, digital art)")
@click.option("--num", "-n", type=int, help="Number of illustrations to generate")
@click.option("--config", "-c", help="Path to configuration file")
@click.option("--output", "-o", help="Output directory or file path")
@click.option("--format", "-f", type=click.Choice(["html", "json", "pdf"]), default="html", help="Output format")
def illustrate(story, style, num, config, output, format):
    """Generate illustrations for a story and create an illustrated storybook."""
    try:
        # Load configuration
        config_dict = load_config(config)
        
        # Load the story
        from aitale.core.models import Story
        click.echo(f"Loading story from {story}...")
        story_obj = Story.load(story)
        
        # Initialize illustration integrator
        illustrator = IllustrationIntegrator(config=config_dict)
        
        # Generate illustrations
        click.echo("Generating illustrations...")
        illustrated_story = illustrator.illustrate(
            story_obj,
            style=style,
            num_illustrations=num
        )
        
        # Export the illustrated story
        if output:
            click.echo(f"Exporting illustrated story as {format}...")
            output_path = illustrated_story.export(output, format=format)
            click.echo(f"Illustrated story exported to {output_path}")
        else:
            # Create a default output path
            default_output = f"illustrated_story_{int(time.time())}"
            output_path = illustrated_story.export(default_output, format=format)
            click.echo(f"Illustrated story exported to {output_path}")
        
        click.echo("Illustration process completed successfully.")
        
    except Exception as e:
        logger.error(f"Error illustrating story: {e}", exc_info=True)
        click.echo(f"Error: {str(e)}")
        sys.exit(1)


@main.command()
@click.option("--output", "-o", required=True, help="Output path for the configuration file")
def init_config(output):
    """Create a default configuration file."""
    try:
        create_default_config(output)
        click.echo(f"Default configuration created at {output}")
        click.echo("\nImportant: You need to edit this file to add your API keys for LLM and image generation services.")
        
    except Exception as e:
        logger.error(f"Error creating configuration: {e}", exc_info=True)
        click.echo(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()