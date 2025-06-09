"""
Training utilities for fall detection models.
"""

from .train_model import train_model
from .utils import plot_training_history

__all__ = ['train_model', 'plot_training_history']
