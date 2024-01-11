# AI Tale Core

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ai-tale/core/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

## Core Engine for AI Tale

AI Tale Core is the central engine that powers [AI Tale](https://aitale.tech/) - an innovative platform that generates illustrated fairy tales using large language models. This repository contains the core components for story generation and illustration integration.

## 🌟 Features

- **Advanced Story Generation**: Leverages state-of-the-art language models to create engaging, age-appropriate fairy tales
- **Illustration Integration**: Seamlessly connects with image generation services to produce illustrations for stories
- **Customizable Themes**: Supports various themes, styles, and cultural contexts for diverse storytelling
- **Extensible Architecture**: Modular design allows for easy integration with other AI Tale components
- **Simple API**: Clean interfaces for integration with web applications and other services

## 📋 Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/ai-tale/core.git
cd core

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Configuration

Copy the example configuration file and modify it according to your needs:

```bash
cp config/config.example.yaml config/config.yaml
```

You'll need to set up API keys for the language model and illustration services in the configuration file.

## 💻 Usage

### Basic Example

```python
from aitale.core import StoryGenerator, IllustrationIntegrator

# Initialize the story generator with configuration
story_gen = StoryGenerator(config_path="config/config.yaml")

# Generate a story
story = story_gen.generate(
    theme="adventure",
    age_group="6-8",
    protagonist="dragon",
    length="medium"
)

# Generate illustrations for the story
illustrator = IllustrationIntegrator(config_path="config/config.yaml")
illustrated_story = illustrator.illustrate(story)

# Export the illustrated story
illustrated_story.export("my_fairy_tale", format="html")
```

### Using the CLI

```bash
python -m aitale.cli generate --theme "magical forest" --age "4-6" --output "my_story.json"
python -m aitale.cli illustrate --story "my_story.json" --output "my_illustrated_story"
```

## 🧩 Architecture

AI Tale Core consists of several key components:

- **Story Generator**: Interfaces with language models to create coherent, engaging stories
- **Illustration Integrator**: Connects with image generation services to create illustrations
- **Content Validator**: Ensures content is appropriate and meets quality standards
- **Export Engine**: Converts stories into various formats (JSON, HTML, PDF)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [AI Tale Website](https://aitale.tech/)
- [Documentation](https://github.com/ai-tale/core/docs)
- [Issue Tracker](https://github.com/ai-tale/core/issues)

## 👥 About DreamerAI

AI Tale is developed by DreamerAI, a Silicon Valley startup focused on AI-related projects, especially applications of large language models and AI-generated content. Founded by Alexander Monash ([GitHub](https://github.com/morfun95)), DreamerAI is committed to creating innovative AI solutions that inspire creativity and imagination.