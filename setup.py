#!/usr/bin/env python3
"""
Setup script for Market Intelligence System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="market-intelligence-system",
    version="1.0.0",
    author="Ajay VNKT",
    author_email="ajayvnkt@example.com",
    description="A comprehensive market intelligence platform for stock analysis and recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ajayvnkt/market-intelligence-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pytest>=7.4.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
        "web": [
            "streamlit>=1.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "market-intelligence=src.comprehensive_market_intelligence:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
