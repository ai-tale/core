"""Image Provider module for AI Tale.

This module defines the interface for image generation providers and
implements specific providers for different image generation services.
"""

import logging
import os
import base64
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple
from io import BytesIO

logger = logging.getLogger(__name__)


class ImageProvider(ABC):
    """Abstract base class for image generation providers.

    This class defines the interface that all image providers must implement.
    It provides methods for generating images and handling provider-specific
    configuration and authentication.
    """

    @abstractmethod
    def generate_image(self, prompt: str, size: Optional[Tuple[int, int]] = None, **kwargs) -> Union[bytes, str]:
        """Generate an image based on the provided prompt.

        Args:
            prompt: The text prompt describing the image to generate.
            size: Tuple of (width, height) for the image size.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Image data as bytes or a string reference to the image.
        """
        pass


class OpenAIImageProvider(ImageProvider):
    """OpenAI DALL-E image generation provider.

    This class implements the ImageProvider interface for OpenAI's DALL-E models,
    which generate images from text descriptions.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the OpenAI image provider.

        Args:
            config: Configuration dictionary with OpenAI-specific settings.
        """
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI package is not installed. Install it with 'pip install openai'.")

        self.config = config
        
        # Set up API key
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Provide it in the config or set the OPENAI_API_KEY environment variable.")
        
        # Initialize the client
        self.client = openai.OpenAI(api_key=api_key)
        
        # Set default model
        self.default_model = config.get("model", "dall-e-3")
        
        # Set default parameters
        self.default_params = {
            "n": config.get("n", 1),
            "size": config.get("size", "1024x1024"),
            "quality": config.get("quality", "standard"),
            "style": config.get("style", "vivid"),
        }

    def generate_image(self, prompt: str, size: Optional[Tuple[int, int]] = None, **kwargs) -> Union[bytes, str]:
        """Generate an image using OpenAI's DALL-E models.

        Args:
            prompt: The text prompt describing the image to generate.
            size: Tuple of (width, height) for the image size.
            **kwargs: Additional OpenAI-specific parameters.

        Returns:
            Image data as a URL string or base64-encoded data.
        """
        # Prepare parameters
        params = self.default_params.copy()
        
        # Override size if provided
        if size:
            width, height = size
            # Convert to OpenAI's size format (must be one of their supported sizes)
            supported_sizes = ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
            requested_size = f"{width}x{height}"
            
            if requested_size in supported_sizes:
                params["size"] = requested_size
            else:
                # Find the closest supported size
                logger.warning(f"Requested size {requested_size} not supported. Using closest supported size.")
                if width >= height:
                    if width >= 1792:
                        params["size"] = "1792x1024"
                    elif width >= 1024:
                        params["size"] = "1024x1024"
                    elif width >= 512:
                        params["size"] = "512x512"
                    else:
                        params["size"] = "256x256"
                else:  # height > width
                    if height >= 1792:
                        params["size"] = "1024x1792"
                    elif height >= 1024:
                        params["size"] = "1024x1024"
                    elif height >= 512:
                        params["size"] = "512x512"
                    else:
                        params["size"] = "256x256"
        
        # Update with any additional parameters
        params.update(kwargs)
        
        try:
            response = self.client.images.generate(
                model=kwargs.get("model", self.default_model),
                prompt=prompt,
                **params
            )
            
            # Get the image URL or base64 data
            image_result = response.data[0]
            
            # Return URL by default
            if hasattr(image_result, 'url') and image_result.url:
                return image_result.url
            
            # Fall back to base64 if URL is not available
            if hasattr(image_result, 'b64_json') and image_result.b64_json:
                return f"data:image/png;base64,{image_result.b64_json}"
            
            raise ValueError("No image data returned from OpenAI API")
            
        except Exception as e:
            logger.error(f"Error generating image with OpenAI: {e}")
            # Return a placeholder image or error message
            return self._generate_error_image(str(e))

    def _generate_error_image(self, error_message: str) -> str:
        """Generate a simple error image with the error message.

        Args:
            error_message: The error message to display.

        Returns:
            Base64-encoded PNG image data as a data URL.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            
            # Create a simple image with the error message
            img = Image.new('RGB', (512, 512), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            
            # Try to use a default font, fall back to default if not available
            try:
                font = ImageFont.truetype("Arial", 20)
            except IOError:
                font = ImageFont.load_default()
            
            # Draw the error message
            d.text((20, 20), "Image Generation Error:", fill=(255, 0, 0), font=font)
            
            # Wrap the error message to fit the image width
            words = error_message.split()
            lines = []
            line = []
            for word in words:
                if len(' '.join(line + [word])) <= 60:  # Adjust based on image width
                    line.append(word)
                else:
                    lines.append(' '.join(line))
                    line = [word]
            if line:
                lines.append(' '.join(line))
            
            # Draw each line of the error message
            y_position = 60
            for line in lines:
                d.text((20, y_position), line, fill=(0, 0, 0), font=font)
                y_position += 30
            
            # Convert the image to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Error generating error image: {e}")
            # Return a simple data URL for a 1x1 transparent pixel
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


class StabilityAIProvider(ImageProvider):
    """Stability AI image generation provider.

    This class implements the ImageProvider interface for Stability AI's models,
    such as Stable Diffusion, which generate images from text descriptions.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Stability AI provider.

        Args:
            config: Configuration dictionary with Stability AI-specific settings.
        """
        self.config = config
        
        # Set up API key
        api_key = config.get("api_key") or os.environ.get("STABILITY_API_KEY")
        if not api_key:
            raise ValueError("Stability AI API key not found. Provide it in the config or set the STABILITY_API_KEY environment variable.")
        
        self.api_key = api_key
        self.api_host = config.get("api_host", "https://api.stability.ai")
        
        # Set default model
        self.default_engine = config.get("engine", "stable-diffusion-xl-1024-v1-0")
        
        # Set default parameters
        self.default_params = {
            "width": config.get("width", 1024),
            "height": config.get("height", 1024),
            "cfg_scale": config.get("cfg_scale", 7.0),
            "samples": config.get("samples", 1),
            "steps": config.get("steps", 30),
        }

    def generate_image(self, prompt: str, size: Optional[Tuple[int, int]] = None, **kwargs) -> Union[bytes, str]:
        """Generate an image using Stability AI's models.

        Args:
            prompt: The text prompt describing the image to generate.
            size: Tuple of (width, height) for the image size.
            **kwargs: Additional Stability AI-specific parameters.

        Returns:
            Image data as bytes or a base64-encoded data URL.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("Requests package is not installed. Install it with 'pip install requests'.")

        # Prepare parameters
        params = self.default_params.copy()
        
        # Override size if provided
        if size:
            params["width"], params["height"] = size
        
        # Update with any additional parameters
        params.update(kwargs)
        
        # Prepare the API request
        engine_id = kwargs.get("engine", self.default_engine)
        
        # Construct the request payload
        payload = {
            "text_prompts": [
                {
                    "text": prompt
                }
            ],
            "cfg_scale": params.get("cfg_scale", 7.0),
            "width": params.get("width", 1024),
            "height": params.get("height", 1024),
            "samples": params.get("samples", 1),
            "steps": params.get("steps", 30),
        }
        
        # Add negative prompt if provided
        if "negative_prompt" in kwargs:
            payload["text_prompts"].append({
                "text": kwargs["negative_prompt"],
                "weight": -1.0
            })
        
        try:
            response = requests.post(
                f"{self.api_host}/v1/generation/{engine_id}/text-to-image",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Non-200 response: {response.text}")
                
            data = response.json()
            
            # Get the first image
            if "artifacts" in data and len(data["artifacts"]) > 0:
                image_data_base64 = data["artifacts"][0]["base64"]
                return f"data:image/png;base64,{image_data_base64}"
            
            raise ValueError("No image data returned from Stability AI API")
            
        except Exception as e:
            logger.error(f"Error generating image with Stability AI: {e}")
            # Return a placeholder image or error message
            return self._generate_error_image(str(e))

    def _generate_error_image(self, error_message: str) -> str:
        """Generate a simple error image with the error message.

        Args:
            error_message: The error message to display.

        Returns:
            Base64-encoded PNG image data as a data URL.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            
            # Create a simple image with the error message
            img = Image.new('RGB', (512, 512), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            
            # Try to use a default font, fall back to default if not available
            try:
                font = ImageFont.truetype("Arial", 20)
            except IOError:
                font = ImageFont.load_default()
            
            # Draw the error message
            d.text((20, 20), "Image Generation Error:", fill=(255, 0, 0), font=font)
            
            # Wrap the error message to fit the image width
            words = error_message.split()
            lines = []
            line = []
            for word in words:
                if len(' '.join(line + [word])) <= 60:  # Adjust based on image width
                    line.append(word)
                else:
                    lines.append(' '.join(line))
                    line = [word]
            if line:
                lines.append(' '.join(line))
            
            # Draw each line of the error message
            y_position = 60
            for line in lines:
                d.text((20, y_position), line, fill=(0, 0, 0), font=font)
                y_position += 30
            
            # Convert the image to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Error generating error image: {e}")
            # Return a simple data URL for a 1x1 transparent pixel
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


class MockImageProvider(ImageProvider):
    """Mock image provider for testing.

    This class implements a simple mock provider that returns predefined
    images for testing purposes without requiring API access.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the mock image provider.

        Args:
            config: Configuration dictionary with mock provider settings.
        """
        self.config = config
        self.image_dir = config.get("image_dir", "")
        self.default_image = config.get("default_image", "")
        
        # Load predefined images if specified
        self.images = {}
        if "images" in config:
            self.images = config["images"]

    def generate_image(self, prompt: str, size: Optional[Tuple[int, int]] = None, **kwargs) -> Union[bytes, str]:
        """Generate a mock image based on the provided prompt.

        Args:
            prompt: The text prompt to match against predefined images.
            size: Ignored in the mock provider.
            **kwargs: Ignored in the mock provider.

        Returns:
            Image data as a file path or base64-encoded data URL.
        """
        # Check if we have a matching image for keywords in this prompt
        for keyword, image_path in self.images.items():
            if keyword.lower() in prompt.lower():
                # If it's a relative path and image_dir is set, make it absolute
                if self.image_dir and not os.path.isabs(image_path):
                    image_path = os.path.join(self.image_dir, image_path)
                
                if os.path.exists(image_path):
                    return image_path
        
        # Return default image if available
        if self.default_image:
            default_path = self.default_image
            if self.image_dir and not os.path.isabs(default_path):
                default_path = os.path.join(self.image_dir, default_path)
            
            if os.path.exists(default_path):
                return default_path
        
        # Generate a simple colored image with text if no default image
        return self._generate_placeholder_image(prompt)

    def _generate_placeholder_image(self, prompt: str) -> str:
        """Generate a simple placeholder image with text from the prompt.

        Args:
            prompt: The text prompt to include in the image.

        Returns:
            Base64-encoded PNG image data as a data URL.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            import hashlib
            
            # Generate a color based on the hash of the prompt
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            r = int(prompt_hash[:2], 16)
            g = int(prompt_hash[2:4], 16)
            b = int(prompt_hash[4:6], 16)
            
            # Create a colored image
            img = Image.new('RGB', (512, 512), color=(r, g, b))
            d = ImageDraw.Draw(img)
            
            # Try to use a default font, fall back to default if not available
            try:
                font = ImageFont.truetype("Arial", 20)
            except IOError:
                font = ImageFont.load_default()
            
            # Draw a title
            d.text((20, 20), "Mock Image Generator", fill=(255, 255, 255), font=font)
            
            # Draw the prompt (truncated if too long)
            short_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
            
            # Wrap the prompt to fit the image width
            words = short_prompt.split()
            lines = []
            line = []
            for word in words:
                if len(' '.join(line + [word])) <= 60:  # Adjust based on image width
                    line.append(word)
                else:
                    lines.append(' '.join(line))
                    line = [word]
            if line:
                lines.append(' '.join(line))
            
            # Draw each line of the prompt
            y_position = 60
            for line in lines:
                d.text((20, y_position), line, fill=(255, 255, 255), font=font)
                y_position += 30
            
            # Convert the image to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Error generating placeholder image: {e}")
            # Return a simple data URL for a 1x1 colored pixel
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


def get_image_provider(config: Dict[str, Any]) -> ImageProvider:
    """Factory function to create an image provider based on configuration.

    Args:
        config: Configuration dictionary with provider settings.

    Returns:
        An instance of a class implementing the ImageProvider interface.

    Raises:
        ValueError: If the specified provider is not supported.
    """
    provider_type = config.get("provider", "openai").lower()
    
    if provider_type == "openai":
        return OpenAIImageProvider(config)
    elif provider_type == "stability":
        return StabilityAIProvider(config)
    elif provider_type == "mock":
        return MockImageProvider(config)
    else:
        raise ValueError(f"Unsupported image provider: {provider_type}")