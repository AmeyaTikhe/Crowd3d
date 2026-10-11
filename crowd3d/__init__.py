"""
crowd3d package initialization.
Re-exports all functionality from crowd3d_all.py.
"""
from __future__ import annotations

import sys
from . import crowd3d_all
from .crowd3d_all import *

# Expose main for root main.py and CLI runners
main = crowd3d_all.main

# Forward compatibility: alias virtual submodules to crowd3d_all so that imports like
# `from crowd3d.geometry import Camera`, `from crowd3d.config import ...`, etc. continue working
for _submod in ['geometry', 'config', 'detection', 'reconstruction', 'counting', 'evaluation', 'main']:
    sys.modules[f'crowd3d.{_submod}'] = crowd3d_all

if __name__ == "__main__":
    main()
