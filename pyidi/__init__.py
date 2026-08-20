__version__ = "1.4.0"

# This has to run before anything imports numba, so it comes before every other
# import in the package. pyMRAW launches numba's thread pool at import time,
# and the threading layer cannot be changed once the pool is up. Left alone it
# resolves to whatever is available: TBB where that is installed, but OpenMP
# where it is not, and GNU OpenMP kills any child forked from a process that
# has used it, which breaks ``processes > 1``. Asking for a fork-safe layer
# here gets in before pyMRAW; ``methods/_lk_kernels.py`` asks again for the
# case where numba was already imported before pyidi.
import os as _os
import sys as _sys

#: True when the threading layer came from the environment rather than from us.
_THREADING_LAYER_FROM_ENV = 'NUMBA_THREADING_LAYER' in _os.environ

if not _THREADING_LAYER_FROM_ENV:
    _os.environ['NUMBA_THREADING_LAYER'] = 'forksafe'
    if 'numba' in _sys.modules:
        # Something imported numba before pyidi (a pytest plugin, say), so it
        # has already read the environment. Setting the live config still
        # counts, as long as the thread pool has not been started yet.
        _sys.modules['numba'].config.THREADING_LAYER = 'forksafe'

# from .pyidi import *
from .pyidi_legacy import pyIDI
from . import tools
from . import postprocessing
from .load_analysis import load_analysis
from .video_reader import VideoReader
from .methods import *
from .GUIs import *
from .fiducial import *
