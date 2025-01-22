from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aitale-core",
    version="0.1.0",
    author="Alexander Monash",
    author_email="alex@dreamerai.com",
    description="Core engine for AI Tale - generates fairy tales using large language models with illustration integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai-tale/core",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "openai>=1.0.0",
        "jinja2>=3.1.2",
        "pillow>=9.0.0",
        "pydantic>=2.0.0",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "aitale=aitale.cli:main",
        ],
    },
    include_package_data=True,
)