"""
Models package for fall detection.
"""

from .fall_detection_gru import FallDetectionGRU
from .fall_detection_lstm import FallDetectionLSTM

__all__ = ['FallDetectionGRU', 'FallDetectionLSTM']
