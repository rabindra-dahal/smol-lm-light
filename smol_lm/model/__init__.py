"""
Package interface router layer.
Exposes the core SmolLM assembly block module up to outer project environments, 
masking underlying sub-package file paths to enforce clean encapsulation rules.
"""
from .transformer import SmolLM

__all__ = ["SmolLM"]
