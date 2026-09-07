"""
Setup script for medix-agent-swarm
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="medix-agent-swarm",
    version="0.1.0",
    author="MediX Team",
    description="Medical assistant with a supervisor, specialist agents, and layered memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qhzeng-gittec/medical-agent-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
