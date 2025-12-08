"""
Utility modules shared by experimental training scripts.

Current contents
----------------
confidence : Tools for tracking EMA confidence heatmaps per case.
morphology : Light-weight morphological operators implemented with pooling.
"""

from .confidence import ConfidenceTracker
from .morphology import create_narrow_band

__all__ = [
    "ConfidenceTracker",
    "create_narrow_band",
]
