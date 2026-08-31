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
    # Publish only the current ``agent`` package; an old local compatibility
    # directory must not become part of new Coding Agent installations.
    packages=find_packages(
        where="src",
        include=["agent", "agent.*"],
        exclude=["guarded_agent", "guarded_agent.*"],
    ),
    python_requires=">=3.10",
    install_requires=["fastapi>=0.115.0", "openai>=1.66.0", "uvicorn>=0.30.0"],
    # The local React bundle is served directly by ``agent.web.app`` after a
    # normal wheel installation, so it must travel with the Python package.
    package_data={"agent.web": ["static/*.html", "static/assets/*.js", "static/assets/*.css"]},
    entry_points={"console_scripts": ["coding-agent=agent.cli:main"]},
)
