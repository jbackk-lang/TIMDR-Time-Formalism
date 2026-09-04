"""TIMDR-Time-Formalism — pełny moduł Chronoprocesu Ξ=(T,x,Γ,φ).

Orkiestruje trzy już zbudowane, niezależne repo-siostry (M/S, G, K) na
wspólnym nośniku T, bez żadnej identyfikacji między nimi. Patrz
timdr_time/chronoprocess.py.
"""

from .chronoprocess import Chronoprocess
from .fourier_bridge import (
    fft_modalities,
    temporal_spread,
    spectral_spread,
    time_bandwidth_product,
)

__all__ = [
    "Chronoprocess",
    "fft_modalities",
    "temporal_spread",
    "spectral_spread",
    "time_bandwidth_product",
]
