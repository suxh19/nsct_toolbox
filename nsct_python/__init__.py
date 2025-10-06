"""
NSCT (Nonsubsampled Contourlet Transform) Toolbox - Python Implementation

This package provides a Python translation of the MATLAB NSCT toolbox.
"""

from nsct_python.core import nssfbdec, nssfbrec, nsfbdec
from nsct_python.filters import (
    efilter2, dmaxflat, ldfilter, ld2quin, mctrans, 
    dfilters, atrousfilters, parafilters
)
from nsct_python.utils import (
    extend2, upsample2df, modulate2, resampz, qupz, symext
)

__all__ = [
    # Core functions
    'nssfbdec', 'nssfbrec', 'nsfbdec',
    # Filter functions
    'efilter2', 'dmaxflat', 'ldfilter', 'ld2quin', 'mctrans',
    'dfilters', 'atrousfilters', 'parafilters',
    # Utility functions
    'extend2', 'upsample2df', 'modulate2', 'resampz', 'qupz', 'symext',
]

__version__ = '0.1.0'