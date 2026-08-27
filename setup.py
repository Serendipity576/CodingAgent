"""Compatibility build configuration for environments with setuptools < 61.

Modern installers read project metadata from ``pyproject.toml``. This fallback
keeps the package and its console entry point installable in older environments
that do not support PEP 621 metadata.
"""

from setuptools import find_packages, setup


setup(
    name="coding-agent",
    version="0.1.0",
    description="A coding agent with workspace-bounded, policy-controlled local tools.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    entry_points={"console_scripts": ["coding-agent=agent.cli:main"]},
)
