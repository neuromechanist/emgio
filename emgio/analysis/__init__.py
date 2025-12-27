# Signal analysis module

from .signal import (
    analyze_signal,
    determine_format_suitability,
    quantization_analysis,
    # Add other signal analysis functions if needed
)
from .verification import compare_signals

__all__ = [
    "analyze_signal",
    "determine_format_suitability",
    "quantization_analysis",
    "compare_signals",  # Add to __all__
]
