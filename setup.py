# setup.py
from setuptools import setup, find_packages

setup(
    name="train-network-rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.1",
        "numpy>=1.24.0",
        "plotly>=5.18.0",
        "typing-extensions>=4.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.1",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
    },
    author="Umar Aslamn",
    author_email="umaraslam66@hotmai.com",
    description="A flexible RL environment for train network optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/umaraslam66/train-network-rl",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)