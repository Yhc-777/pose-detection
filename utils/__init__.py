"""
Utilities package for pose detection and fall detection.
"""

from .body import BODY_PARTS_NAMES, BODY_CONNECTIONS_DRAW, BODY_CONNECTIONS, BODY_GROUPS, draw_skeleton
from .extract_keypoints_from_video import extract_keypoints_from_video

__all__ = [
    'BODY_PARTS_NAMES',
    'BODY_CONNECTIONS_DRAW', 
    'BODY_CONNECTIONS',
    'BODY_GROUPS',
    'draw_skeleton',
    'extract_keypoints_from_video'
]
